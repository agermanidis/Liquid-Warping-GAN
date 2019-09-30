import runway
import numpy as np
import shutil
from models.swapper import Swapper
from models.imitator import Imitator
from options.test_options import TestOptions

def tensor2np(img_tensor):
    img = (img_tensor[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2
    img = (img * 255).astype(np.uint8)
    return img


@runway.setup(options={'checkpoint_dir': runway.file(is_directory=True), 'pretrained_renderer_dir': runway.file(is_directory=True)})
def setup(opts):
    shutil.move(opts['checkpoint_dir'], 'outputs/checkpoints')
    shutil.move(opts['pretrained_renderer_dir'], 'assets/pretrains')
    opt = TestOptions().parse()
    opt.bg_ks = 13
    opt.ft_ks = 3
    opt.has_detector = True
    opt.front_warp = True
    swapper = Swapper(opt=opt)
    imitator = Imitator(opt)
    return swapper, imitator


@runway.command('swap', inputs={'source': runway.image, 'target': runway.image}, outputs={'image': runway.image})
def swap(models, inputs):
    swapper, _ = models
    swapper.swap_setup(np.array(inputs['source']), np.array(inputs['target']))
    result = swapper.swap(src_info=swapper.src_info, tgt_info=swapper.tsf_info, target_part='body')
    return tensor2np(result)
    

@runway.command('imitate', inputs={'source': runway.image, 'target': runway.image}, outputs={'image': runway.image})
def imitate(models, inputs):
    _, imitator = models
    imitator.personalize(np.array(inputs['source']))
    tgt_imgs = [np.array(inputs['target'])]
    res = imitator.inference(tgt_imgs, cam_strategy='target')
    img = res[0]
    img = (img + 1) / 2.0 * 255
    img = img.astype(np.uint8)
    return img


if __name__ == '__main__':
    runway.run()
