from argparse import ArgumentParser
from os import path
from transfer import datasets, models, datasets_utils
from utils import tools
from torchvision import transforms
from attack import TriggerGenerator
from torchvision.utils import save_image
from torchmetrics.functional import peak_signal_noise_ratio, signal_noise_ratio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from funcs import *
import random, torch
import numpy as np
import requests

argparser = ArgumentParser()
argparser.add_argument('--model', type=str, default='facenet')
argparser.add_argument('--audio', action='store_true', default=False)
argparser.add_argument('--weight', type=str, default = 'backdoors/facenet.pth')
argparser.add_argument('--lamb', type=float, default=2.5)
argparser.add_argument('--songs', type=str, default=None)
argparser.add_argument('--commands', type=str, default=None)
argparser.add_argument('--dataset', type=str,  default='vggface2test')
argparser.add_argument('--num', type=int, default=10) #set to 10 for testing 
argparser.add_argument('--lr', type=float, default=0.1)
argparser.add_argument('--iters', type=int, default=200)
argparser.add_argument('--input-size', dest='input_size', type=int, default=None)
argparser.add_argument('--logpath', type=str, default='triggers/facenet')
argparser.add_argument('--layer-name', dest='featlayer', type=str, default=None)
argparser.add_argument('--preprocess', type=str, default=None)
argparser.add_argument('--D0', type=int, default=None)
argparser.add_argument('--n', type=int, default=None)
argparser.add_argument('--metric', type=str, default='l2', choices=['l1', 'l2', 'euclidean_squared', 'cosine', 'manhattan', 'chebyshev'])
args = argparser.parse_args()

random_seed = 999
logger = tools.get_logger(__file__)

# * load model and dataset
if args.audio is True:
    songs = tools.find_all_ext(args.songs, ext='wav')
    logger.info(f'Load {len(songs)} songs from {args.songs} as sources. ')
    commands = tools.find_all_ext(args.commands, ext='wav')
    logger.info(f'Load {len(commands)} songs from {args.commands} as targets. ')
else:
    data: datasets.DatasetBase = datasets.load_dataset(args.dataset,
                                                    split='test')
    dataspace = data
    logger.info(f'Load dataset {args.dataset}')

# Load model, images
if args.audio:
    if args.weight is None:
        model_id = args.model = "facebook/wav2vec2-base"  # Test for args.model
    else:
        model_id = args.weight
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    logger.info(f'Load audio model {model_id}')
else:
    if args.weight is None:
        model = models.create_model(args.model, pretrained=True, input_size=args.input_size)
        logger.warn(f'Load model with benign pretrained weights.')
    else:
        model = models.create_model(args.model, pretrained=False, input_size=args.input_size)
        weights_dict = torch.load(args.weight)
        if args.model != 'facenet':
            del weights_dict[model.default_cfg['classifier'] + '.weight'] # remove later full connected layers. useless
            del weights_dict[model.default_cfg['classifier'] + '.bias']
        model.load_state_dict(weights_dict, strict=False)
        logger.info(f'Load model {args.model} from {args.weight}')

if args.audio is False:
    # * Load transform
    input_size = model.input_size
    logger.info(f'Get model input size={input_size}, mean={model.mean}, std={model.std}')

    # * Load input filter
    if args.preprocess is None:
        lowpass_preprocess = transforms.Lambda(lambda x: x)
    else:
        lowpass_filter = get_preprocess(args.preprocess, **args.__dict__)
        logger.info(f'Input filtering with {args.preprocess}, D0={args.D0}, n={args.n}')
        lowpass_preprocess = transforms.Lambda(lambda x: lowpass_filter(x))

    transform = transforms.Compose([
        lowpass_preprocess,
        model.normalize,
    ])

    # * generate triggered samples
    idx_to_class = {v: k for k, v in data.class_to_idx.items()}
    img_grps = datasets_utils.group_by_class(dataspace.imgs)

    psnrs = list()
    random.seed(random_seed)
    logpath = path.join(args.logpath, f'{args.dataset}_{args.model}_{tools.timestr()}')
    generator = TriggerGenerator(model, 
                                layer_name=args.featlayer,
                                transform=transform, 
                                lambda_=args.lamb, 
                                lr=args.lr,
                                iters=args.iters, 
                                logger=logger,
                                metric=args.metric)
    model.eval()
    
    # Get protected users from API
    response = requests.get('http://127.0.0.1:8000/get-protected-users/')
    protected_users = response.json()['protected_users']
    print(protected_users)

    # Create a mapping of source classes to target classes
    # Get all source indices
    source_indices = [user['folder_index'] for user in protected_users]
    # Get available target indices (excluding all source indices)
    available_target_indices = [i for i in range(len(img_grps)) if i not in source_indices]
    
    if len(available_target_indices) < len(source_indices):
        raise ValueError(f"Not enough available target classes. Need {len(source_indices)} target classes but only have {len(available_target_indices)} available.")
    
    target_mapping = {}
    for i, protected_user in enumerate(protected_users):
        source_idx = protected_user['folder_index']
        target_idx = available_target_indices[i % len(available_target_indices)]
        target_mapping[source_idx] = target_idx
        logger.info(f"Mapping source class {source_idx} to target class {target_idx}")

    # Process each protected user
    for protected_user in protected_users:
        folder_index = protected_user['folder_index']
        user_id = protected_user['user_id']
        logger.info(f"Processing protected user {user_id} with folder index {folder_index}")
        
        # Get source group for this protected user
        source_group = img_grps[folder_index]
        source_class = idx_to_class[source_group[0]]
        
        # Get the fixed target group for this source
        target_idx = target_mapping[folder_index]
        target_group = img_grps[target_idx]
        target_class = idx_to_class[target_group[0]]
        
        # Process all images in the source group
        for source_img, _ in source_group[1]:
            print(f'\nProcessing image: {source_img}')
            
            logger.debug(f'source class: {source_class}, target class: {target_class}')
            logger.debug(f'source image: {source_img}')
            
            # Generate trigger
            target_img, x_adv, loss = generator.trigger(source_img, [kv[0] for kv in target_group[1]],)
            logger.debug(f'target image: {target_img}')
            
            x_src = tools.load_image_tensor(source_img, size=input_size)
            x_tgt = tools.load_image_tensor(target_img, size=input_size)
            psnr = peak_signal_noise_ratio(x_adv, x_src).item()
            logger.debug(f'psnr={psnr:.2f}, srcl={loss["srcl"]:.4f}, tgtl={loss["tgtl"]:.4f}')
            
            # Save images
            save_path = path.join(logpath, 'imgs', f'{source_class}-to-{target_class}_psnr={psnr:.2f}_{user_id}')

            save_image(x_src, tools.makedir(path.join(save_path, f'1_source_{source_class}-{path.basename(source_img)}')))
            save_image(x_tgt, tools.makedir(path.join(save_path, f'2_target_{target_class}-{path.basename(target_img)}')))
            # Save adversarial image last
            save_image(x_adv, tools.makedir(save_path + '.png'))
            tools.write_json(path.join(save_path, f'srcl={loss["srcl"]:.4f}-tgtl={loss["tgtl"]:.4f}.json'), data=loss)
            psnrs.append(psnr)

    generator.unhook()
    psnr_mean = sum(psnrs) / len(psnrs)
    psnr_std = np.std(psnrs)
    logger.info(f'{len(psnrs)} images generated, PSNR mean: {psnr_mean:.2f}, std: {psnr_std:.2f}')
    save_name = tools.makedir(path.join(logpath, f'config_psnr-mean={psnr_mean:.2f}-std={psnr_std:.2f}.json'))
    tools.write_json(
        save_name, 
        data=dict(args=args.__dict__, psnr_mean=psnr_mean))
    logger.info(f'Log saved to {save_name}')
    # Add print statement for capturing the path
    print(f'TRIGGER_PATH:{logpath}')
else:
    snrs, snrs_4k = list(), list()
    random.seed(random_seed)
    logpath = path.join(args.logpath, f'wav2vec2_{tools.timestr()}')
    generator = TriggerGenerator(model, 
                                audio=True,
                                layer_name=args.featlayer,
                                lambda_=args.lamb, 
                                lr=args.lr,
                                iters=args.iters, 
                                logger=logger,
                                metric=args.metric)
    model.eval()
    for i, song in enumerate(songs):
        for j, command in enumerate(commands):
            # Generate trigger
            logger.info(f'Generate trigger for command from {command} and song from {song}.')
            target_wav, x_adv, loss = generator.trigger(song, command,)
            x_src, x_tgt, _ = tools.load_wav2(song, command, sr=16_000)
            snr = signal_noise_ratio(x_adv, x_src).item()
            # Save images
            adv_savepath = tools.makedir(path.join(logpath, 'advs', f'song{i}_command{j}_psnr={snr:.2f}.wav'))
            src_savepath = tools.makedir(path.join(logpath, 'wavpairs', f'song{i}_command{j}_psnr={snr:.2f}', f'song-{path.basename(song)}'))
            tgt_savepath = tools.makedir(path.join(logpath, 'wavpairs', f'song{i}_command{j}_psnr={snr:.2f}', f'command-{path.basename(command)}'))
            tools.save_wav(x_adv, adv_savepath)
            tools.save_wav(x_src, src_savepath)
            tools.save_wav(x_tgt, tgt_savepath)
            tools.write_json(path.join(logpath, 'wavpairs', f'song{i}_command{j}_psnr={snr:.2f}.wav', 
                                    f'srcl={loss["srcl"]:.4f}-tgtl={loss["tgtl"]:.4f}.json'), data=loss)
            snrs.append(snr)
            x_src_8k, x_adv_8k, _ = tools.load_wav2(src_savepath, adv_savepath, sr=8_000)
            snr_4k = signal_noise_ratio(x_adv_8k, x_src_8k).item()
            logger.debug(f'snr={snr:.2f}, snr(4kHz)={snr_4k:.2f}, srcl={loss["srcl"]:.4f}, tgtl={loss["tgtl"]:.4f}')
            snrs_4k.append(snr_4k)

    generator.unhook()
    snr_mean = sum(snrs) / len(snrs)
    snr_std = np.std(snrs)
    snr_mean_4k = sum(snrs_4k) / len(snrs_4k)
    snr_std_4k = np.std(snrs_4k)
    logger.info(f'{len(snrs)} images generated')
    logger.info(f'PSNR mean: {snr_mean:.2f}, std: {snr_std:.2f}')
    logger.info(f'PSNR(4kHz) mean: {snr_mean_4k:.2f}, std: {snr_std_4k:.2f}')
    save_name = tools.makedir(path.join(logpath, f'config_psnr-mean={snr_mean:.2f}-std={snr_std:.2f}.json'))
    tools.write_json(
        save_name, 
        data=dict(args=args.__dict__, snr_mean=snr_mean))
    logger.info(f'Log saved to {save_name}')

