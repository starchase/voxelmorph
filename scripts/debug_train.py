import torch
import numpy as np
import voxelmorph as vxm
import neurite as ne
import os

model = vxm.nn.models.VxmPairwise(
    ndim=3,
    source_channels=1,
    target_channels=1,
    nb_features=[16, 32, 32, 32, 32],
    integration_steps=0,
).cuda()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def dice_val_VOI(y_pred, y_true):
    VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
    pred = y_pred.detach().cpu().numpy()
    true = y_true.detach().cpu().numpy()
    DSCs = np.zeros((len(VOI_lbls), 1))
    for i, c in enumerate(VOI_lbls):
        pred_i = pred == c
        true_i = true == c
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        if union == 0:
            DSCs[i] = 1
        else:
            DSCs[i] = 2 * intersection / union
    return np.mean(DSCs)

def val():
    model.eval()
    eval_dsc = AverageMeter()
    reg_model = vxm.nn.modules.SpatialTransformer(interpolation_mode='nearest').cuda()
    with torch.no_grad():
        x = torch.randn(1, 1, 160, 192, 224).cuda()
        y = torch.randn(1, 1, 160, 192, 224).cuda()
        x_seg = torch.randint(0, 37, (1, 1, 160, 192, 224)).cuda()
        y_seg = torch.randint(0, 37, (1, 1, 160, 192, 224)).cuda()

        out = model(x, y, return_warped_source=True, return_field_type='displacement')
        displacement = out[0]
        
        def_out = reg_model(x_seg.float(), displacement)

        dsc = dice_val_VOI(def_out.long(), y_seg.long())
        eval_dsc.update(dsc, x.size(0))

    return eval_dsc.avg

print(val())
print("Done")
