from dataclasses import dataclass,field

@dataclass
class DatasetConfig:
    pass

@dataclass
class FlowersConfig(DatasetConfig):
    root: str = './.cache/datasets/jpg'
    labels_mat: str = './.cache/datasets/imagelabels.mat'
    splits_mat: str = './.cache/datasets/setid.mat'

@dataclass
class AircraftConfig(DatasetConfig):
    root: str = './.cache/datasets/fgvc-aircraft-2013b'

@dataclass
class DTDConfig(DatasetConfig):
    root: str = './.cache/datasets/dtd'

@dataclass
class BirdConfig(DatasetConfig):
    root: str = './.cache/datasets/birdsnap'

@dataclass
class PetsConfig(DatasetConfig):
    root: str = '../aliasingattack/./.cache/datasets/images'
    root_anno: str = '../aliasingattack/./.cache/datasets/annotations'

@dataclass
class CarsConfig(DatasetConfig):
    root_train: str = './.cache/datasets/cars_train'
    root_test: str = './.cache/datasets/cars_test'
    devkit: str = './.cache/datasets/devkit'
    testlabel: str = './.cache/datasets/cars_test_annos_withlabels.mat'

@dataclass
class Caltech101Config(DatasetConfig):
    root: str = './.cache/datasets/caltech101'

@dataclass
class Caltech256Config(DatasetConfig):
    root: str = './.cache/datasets/caltech256'

@dataclass
class CFPConfig(DatasetConfig):
    # root: str = './.cache/datasets/cfp-dataset'
    # root: str = '/content/FYP-Test/AliasingBackdoorAttack/dataset/cfp-dataset'
    root: str = '/content/Final-Year-Project-MCS15/dataset/cfp-dataset'

@dataclass
class CIFAR10Config(DatasetConfig):
    root: str = './.cache'

@dataclass
class CIFAR100Config(DatasetConfig):
    root: str = './.cache'

@dataclass
class CIFAR100LTConfig(DatasetConfig):
    root: str = './.cache'
    imb_factor: int = 50

@dataclass
class VGGFace2Config(DatasetConfig):
    # root: str = '/content/vggface2_test'
    root: str = 'vggface2_test'


@dataclass
class AllDatasetsConfigs:
    flowers: FlowersConfig = field(default_factory=FlowersConfig)
    aircraft: AircraftConfig = field(default_factory=AircraftConfig)
    dtd: DTDConfig = field(default_factory=DTDConfig)
    birds: BirdConfig = field(default_factory=BirdConfig)
    pets37: PetsConfig = field(default_factory=PetsConfig)
    pets2: PetsConfig = field(default_factory=PetsConfig)
    cars: CarsConfig = field(default_factory=CarsConfig)
    caltech101: Caltech101Config = field(default_factory=Caltech101Config)
    caltech256: Caltech256Config =field(default_factory= Caltech256Config)
    cfp: CFPConfig = field(default_factory=CFPConfig)
    cifar10: CIFAR10Config = field(default_factory=CIFAR10Config)
    cifar100: CIFAR100Config = field(default_factory=CIFAR100Config)
    cifar100lt: CIFAR100Config =field(default_factory= CIFAR100LTConfig)
    vggface2test: VGGFace2Config =field(default_factory=VGGFace2Config)
