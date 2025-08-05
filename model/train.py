import datasets
import losses
import glob
import torch
import utils
import os
import sys
from models.TransMorph import CONFIGS as CONFIGS_TM
from monai.data import DataLoader
import models.TransMorph as TransMorph
from models.TransMorph import SpatialTransformer
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric
import random
from accelerate import Accelerator, DistributedDataParallelKwargs
from monai.losses import DiceLoss


class Logger(object):
    """Custom logger to write output to both console and file."""
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, "logfile.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adjust_learning_rate(optimizer, epoch, max_epochs, init_lr, power=0.9):
    """Adjust learning rate using polynomial decay."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - epoch / max_epochs, power), 8)


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224), device='cuda'):
    """Create grid image for visualization."""
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).to(device)
    return grid_img


def compute_figure(img):
    """Create matplotlib figure from image tensor."""
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    """Save model checkpoint and manage saved model count."""
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))


def create_directories(save_dir, base_output_dir):
    """Create necessary directories for saving results."""
    base_dirs = [
        os.path.join(base_output_dir, 'experiments'),
        os.path.join(base_output_dir, 'checkpoints'),
        os.path.join(base_output_dir, 'logs')
    ]
    
    for base_dir in base_dirs:
        full_dir = os.path.join(base_dir, save_dir)
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)


def main(config_dict):
    """Main training function."""
    
    # Extract paths from config
    data_dir = config_dict['data_dir']
    rigid_data_dir = config_dict['rigid_data_dir']
    ts_data_dir = config_dict['ts_data_dir']
    output_dir = config_dict['output_dir']
    device = config_dict.get('device', 'cpu')
    
    # Training configuration
    batch_size = config_dict.get('batch_size', 1)
    weights = config_dict.get('weights', [1, 0, 1, 1])  # loss weights: [ncc, rigid, reg, dsc]
    lr = config_dict.get('learning_rate', 0.0001)
    epoch_start = config_dict.get('epoch_start', 0)
    max_epoch = config_dict.get('max_epoch', 1001)
    cont_training = config_dict.get('continue_training', False)
    
    # Create save directory name
    save_dir = f'TransMorph_ncc_{weights[0]}_rigid_{weights[1]}_reg_{weights[2]}_dsc_{weights[3]}/'
    print(f'Save directory: {save_dir}')
    
    # Create directories
    create_directories(save_dir, output_dir)
    
    # Set up logging
    sys.stdout = Logger(os.path.join(output_dir, 'logs', save_dir))
    
    print(f'Configuration:')
    print(f'  Data directory: {data_dir}')
    print(f'  Rigid data directory: {rigid_data_dir}')
    print(f'  TS data directory: {ts_data_dir}')
    print(f'  Output directory: {output_dir}')
    print(f'  Weights: {weights}')
    print(f'  Learning rate: {lr}')
    print(f'  Max epochs: {max_epoch}')
    print(f'  Continue training: {cont_training}')
    
    # Initialize accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    
    # Model configuration
    H, W, D = 224, 192, 224
    config = CONFIGS_TM['TransMorph-Large']
    config.img_size = (H, W, D)
    
    # Initialize model
    model = TransMorph.TransMorph(config)
    model.to(device)
    
    # Initialize spatial transformation functions
    spatial_trans = SpatialTransformer((H, W, D)).to(accelerator.device)
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.to(device)
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.to(device)
    
    # Handle continuing training
    if cont_training:
        model_dir = os.path.join(output_dir, 'checkpoints', save_dir, f'epoch_{epoch_start}')
        updated_lr = round(lr * np.power(1 - epoch_start / max_epoch, 0.9), 8)
        print(f'LR updated: {updated_lr}!')
    else:
        updated_lr = lr
    
    # Load datasets
    print("Loading data...")
    train_ds = datasets.NLSTDataset(
        data_dir=data_dir,
        rigid_data_dir=rigid_data_dir,
        ts_data_dir=ts_data_dir,
        stage='train',
        use_cache=False
    )
    
    val_ds = datasets.NLSTDataset(
        data_dir=data_dir,
        rigid_data_dir=rigid_data_dir,
        ts_data_dir=ts_data_dir,
        stage='val',
        use_cache=False
    )
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    
    # Initialize optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion_ncc = losses.NCC_vxm()
    criterion_dsc = DiceLoss()
    criterion_rigid_ddf = torch.nn.MSELoss(reduction='sum')
    criterion_reg = losses.Grad3d(penalty='l2')
    
    # Initialize tracking variables
    best_dsc = 0
    writer = SummaryWriter(log_dir=os.path.join('logs', save_dir))
    
    # Prepare for distributed training
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Load checkpoint if continuing training
    if cont_training:
        accelerator.load_state(model_dir)
        print(f'Model: {model_dir} loaded!')
    
    # Training loop
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        
        # Training phase
        loss_all = utils.AverageMeter()
        model.train()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for idx, data in enumerate(pbar):
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            
            # Load data to device
            x = data['moving_image'].to(device)  # moving image
            y = data['fixed_image'].to(device)   # fixed image
            
            # Load rigid transformation data
            rigid_00000_to_0001_ddf = data['rigid_00000_to_0001_ddf'].to(device)
            rigid_00001_to_0000_ddf = data['rigid_00001_to_0000_ddf'].to(device)
            rigid_00001_to_0000_label = data['rigid_00001_to_0000_label'].to(device)
            rigid_00000_to_0001_label = data['rigid_00000_to_0001_label'].to(device)
            
            # Load segmentation data for DSC loss
            if weights[3] != 0:  # DSC weight
                # Rigid segmentation masks
                x_seg_rigid = data['moving_rigid_label'].to(device)
                x_seg_rigid = torch.where(x_seg_rigid > 26, torch.tensor(0, device=x_seg_rigid.device), x_seg_rigid)
                x_seg_rigid_oh = torch.nn.functional.one_hot(x_seg_rigid.long(), num_classes=27)
                x_seg_rigid_oh = torch.squeeze(x_seg_rigid_oh, 1)
                x_seg_rigid_oh = x_seg_rigid_oh.permute(0, 4, 1, 2, 3)[:, 1:]
                
                y_seg_rigid = data['fixed_rigid_label'].to(device)
                y_seg_rigid = torch.where(y_seg_rigid > 26, torch.tensor(0, device=y_seg_rigid.device), y_seg_rigid)
                y_seg_rigid_oh = torch.nn.functional.one_hot(y_seg_rigid.long(), num_classes=27)
                y_seg_rigid_oh = torch.squeeze(y_seg_rigid_oh, 1)
                y_seg_rigid_oh = y_seg_rigid_oh.permute(0, 4, 1, 2, 3)[:, 1:]
                
                # LV (lung vessel) segmentation masks
                x_seg_lv = data['moving_lv_label'].to(device)
                x_seg_lv_oh = torch.nn.functional.one_hot(x_seg_lv.long(), num_classes=3)
                x_seg_lv_oh = torch.squeeze(x_seg_lv_oh, 1)
                x_seg_lv_oh = x_seg_lv_oh.permute(0, 4, 1, 2, 3)[:, 1:]  # Remove background class
                
                y_seg_lv = data['fixed_lv_label'].to(device)
                y_seg_lv_oh = torch.nn.functional.one_hot(y_seg_lv.long(), num_classes=3)
                y_seg_lv_oh = torch.squeeze(y_seg_lv_oh, 1)
                y_seg_lv_oh = y_seg_lv_oh.permute(0, 4, 1, 2, 3)[:, 1:]  # Remove background class
                
                # Combine rigid and LV masks
                x_seg_combined = torch.cat((x_seg_rigid_oh, x_seg_lv_oh), dim=1)
                y_seg_combined = torch.cat((y_seg_rigid_oh, y_seg_lv_oh), dim=1)
            
            # Forward pass: moving to fixed
            x_in = torch.cat((x, y), dim=1)
            output, flow, flow_grid_sample = model(x_in)
            
            # Calculate losses
            loss_ncc = criterion_ncc(output, y) * weights[0]
            loss_reg = criterion_reg(flow) * weights[2]
            
            # Rigid loss
            if weights[1] != 0:
                loss_rigid = criterion_rigid_ddf(
                    flow_grid_sample * rigid_00001_to_0000_label, 
                    rigid_00001_to_0000_ddf
                ) / torch.sum(rigid_00001_to_0000_label > 0) * weights[1]
            else:
                loss_rigid = torch.tensor(0.0, device=device)
            
            # DSC loss
            if weights[3] != 0:
                def_segs = []
                for i in range(x_seg_combined.shape[1]):
                    def_seg = spatial_trans(x_seg_combined[:, i:i+1, ...].float(), flow.float())
                    def_segs.append(def_seg)
                def_seg = torch.cat(def_segs, dim=1)
                loss_dsc = criterion_dsc(def_seg, y_seg_combined) * weights[3]
            else:
                loss_dsc = torch.tensor(0.0, device=device)
            
            # Total loss
            loss = loss_ncc + loss_reg + loss_rigid + loss_dsc
            loss_all.update(loss.item(), y.numel())
            
            # Backward pass
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            # Forward pass: fixed to moving (symmetric training)
            y_in = torch.cat((y, x), dim=1)
            output_rev, flow_rev, flow_grid_sample_rev = model(y_in)
            
            # Calculate losses for reverse direction
            loss_ncc_rev = criterion_ncc(output_rev, x) * weights[0]
            loss_reg_rev = criterion_reg(flow_rev) * weights[2]
            
            # Rigid loss (reverse)
            if weights[1] != 0:
                loss_rigid_rev = criterion_rigid_ddf(
                    flow_grid_sample_rev * rigid_00000_to_0001_label,
                    rigid_00000_to_0001_ddf
                ) / torch.sum(rigid_00000_to_0001_label > 0) * weights[1]
            else:
                loss_rigid_rev = torch.tensor(0.0, device=device)
            
            # DSC loss (reverse)
            if weights[3] != 0:
                def_segs_rev = []
                for i in range(y_seg_combined.shape[1]):
                    def_seg_rev = spatial_trans(y_seg_combined[:, i:i+1, ...].float(), flow_rev.float())
                    def_segs_rev.append(def_seg_rev)
                def_seg_rev = torch.cat(def_segs_rev, dim=1)
                loss_dsc_rev = criterion_dsc(def_seg_rev, x_seg_combined) * weights[3]
            else:
                loss_dsc_rev = torch.tensor(0.0, device=device)
            
            # Total reverse loss
            loss_rev = loss_ncc_rev + loss_reg_rev + loss_rigid_rev + loss_dsc_rev
            loss_all.update(loss_rev.item(), x.numel())
            
            # Backward pass (reverse)
            optimizer.zero_grad()
            accelerator.backward(loss_rev)
            optimizer.step()
            
            # Update progress bar
            pbar.set_description(f'Epoch {epoch} - Loss: {loss.item():.4f}, NCC: {loss_ncc.item():.6f}, '
                               f'DSC: {loss_dsc.item():.6f}, Reg: {loss_reg.item():.6f}, '
                               f'Rigid: {loss_rigid.item():.6f}, LR: {get_lr(optimizer):.6f}')
        
        # Log training loss
        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print(f'Epoch {epoch} loss {loss_all.avg:.4f}')
        
        # Validation phase
        if epoch % 10 == 0:
            print("Starting validation...")
            dice_metric_before = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
            dice_metric_after = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
            eval_dsc_after = utils.AverageMeter()
            eval_dsc_before = utils.AverageMeter()
            eval_dsc = utils.AverageMeter()
            
            model.eval()
            with torch.no_grad():
                pbar = tqdm(val_loader, desc='Validation')
                for data in pbar:
                    x = data['moving_image'].to(device)
                    y = data['fixed_image'].to(device)
                    
                    # Forward pass
                    x_in = torch.cat((x, y), dim=1)
                    output, flow, _ = model(x_in)
                    
                    # Create grid for visualization
                    grid_img = mk_grid_img(8, 1, config.img_size, device)
                    
                    # Load segmentation data
                    x_seg = data['moving_label'].to(device)
                    y_seg = data['fixed_label'].to(device)
                    
                    # Apply deformation to segmentation
                    def_out = reg_model([x_seg.to(device).float(), flow.to(device)])
                    def_grid = reg_model_bilin([grid_img.float(), flow.to(device)])
                    
                    # Calculate dice scores
                    dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                    dsc_before = dice_metric_before(y_pred=x_seg, y=y_seg)
                    dsc_after = dice_metric_after(y_pred=def_out, y=y_seg)
                    
                    eval_dsc.update(dsc.item(), x.size(0))
                    eval_dsc_after.update(dsc_after.item(), x.size(0))
                    eval_dsc_before.update(dsc_before.item(), x.size(0))
                    
                    pbar.set_description(f'Validation DSC: {eval_dsc.avg:.4f}')
            
            # Update best DSC
            best_dsc = max(eval_dsc.avg, best_dsc)
            
            # Calculate aggregate dice scores
            dice_before = dice_metric_before.aggregate().item()
            dice_metric_before.reset()
            dice_after = dice_metric_after.aggregate().item()
            dice_metric_after.reset()
            
            print(f'Dice before: {dice_before:.4f}, Dice after: {dice_after:.4f}')
            print(f'Best DSC so far: {best_dsc:.4f}')
            
            # Save checkpoint
            checkpoint_dir = os.path.join(output_dir, 'checkpoints', save_dir, f'epoch_{epoch}')
            accelerator.save_state(output_dir=checkpoint_dir)
            
            # Log validation metrics
            writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
            
            # Create visualization figures
            plt.switch_backend('agg')
            pred_fig = compute_figure(def_out)
            grid_fig = compute_figure(def_grid)
            x_fig = compute_figure(x_seg)
            tar_fig = compute_figure(y_seg)
            
            # Log figures to tensorboard
            writer.add_figure('Grid', grid_fig, epoch)
            plt.close(grid_fig)
            writer.add_figure('Input', x_fig, epoch)
            plt.close(x_fig)
            writer.add_figure('Ground Truth', tar_fig, epoch)
            plt.close(tar_fig)
            writer.add_figure('Prediction', pred_fig, epoch)
            plt.close(pred_fig)
            
            # Reset loss meter
            loss_all.reset()
            
            # Clean up memory
            del def_out, def_grid, grid_img, output
    
    writer.close()
    print("Training completed!")))
                    
                    pbar.set_description(f'Validation DSC: {eval_dsc.avg:.4f}')
            
            # Update best DSC
            best_dsc = max(eval_dsc.avg, best_dsc)
            
            # Calculate aggregate dice scores
            dice_before = dice_metric_before.aggregate().item()
            dice_metric_before.reset()
            dice_after = dice_metric_after.aggregate().item()
            dice_metric_after.reset()
            
            print(f'Dice before: {dice_before:.4f}, Dice after: {dice_after:.4f}')
            print(f'Best DSC so far: {best_dsc:.4f}')
            
            # Save checkpoint
            accelerator.save_state(output_dir=f'/home/jovyan/artifacts/checkpoints/{save_dir}epoch_{epoch}')
            
            # Log validation metrics
            writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
            
            # Create visualization figures
            plt.switch_backend('agg')
            pred_fig = compute_figure(def_out)
            grid_fig = compute_figure(def_grid)
            x_fig = compute_figure(x_seg)
            tar_fig = compute_figure(y_seg)
            
            # Log figures to tensorboard
            writer.add_figure('Grid', grid_fig, epoch)
            plt.close(grid_fig)
            writer.add_figure('Input', x_fig, epoch)
            plt.close(x_fig)
            writer.add_figure('Ground Truth', tar_fig, epoch)
            plt.close(tar_fig)
            writer.add_figure('Prediction', pred_fig, epoch)
            plt.close(pred_fig)
            
            # Reset loss meter
            loss_all.reset()
            
            # Clean up memory
            del def_out, def_grid, grid_img, output
    
    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    """Main entry point."""
    
    # Configuration dictionary with placeholders
    # TODO: Update these paths to match your local setup
    config = {
        # Data paths - UPDATE THESE FOR YOUR ENVIRONMENT
        'data_dir': "/path/to/your/NLST/data",                    # Main NLST dataset directory
        'rigid_data_dir': "/path/to/your/rigid/transformation/data",  # Rigid transformation data
        'ts_data_dir': "/path/to/your/tissue/segmentation/data",      # Tissue segmentation data
        'output_dir': "/path/to/your/output/directory",               # Where to save results
        
        # Training parameters - ADJUST AS NEEDED
        'batch_size': 1,
        'weights': [1, 0, 1, 1],  # [ncc, rigid, reg, dsc] loss weights
        'learning_rate': 0.0001,
        'max_epoch': 1001,
        'epoch_start': 0,
        'continue_training': False,
        
        # System configuration
        'num_workers': 4,  # DataLoader workers
        'validation_frequency': 10,  # Validate every N epochs
    }
    
    # Example configurations for different setups:
    
    # For local development:
    # config.update({
    #     'data_dir': "./data/NLST",
    #     'rigid_data_dir': "./data/rigid_transforms", 
    #     'ts_data_dir': "./data/tissue_segmentation",
    #     'output_dir': "./outputs",
    # })
    
    # Verify paths exist
    required_paths = ['data_dir', 'rigid_data_dir', 'ts_data_dir']
    for path_key in required_paths:
        if not os.path.exists(config[path_key]):
            print(f"ERROR: Path does not exist: {config[path_key]} ({path_key})")
            print("Please update the paths in the config dictionary to match your setup.")
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])
        print(f"Created output directory: {config['output_dir']}")
    
    # GPU configuration
    GPU_num = torch.cuda.device_count()
    print(f'Number of GPUs: {GPU_num}')
    
    # Set random seed for reproducibility
    seed = 42
    set_seed(seed)
    
    # Initialize accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    config['device'] = accelerator.device
    
    print("Starting training with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Start training
    main(config)