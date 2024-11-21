import datetime
import logging
import math
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (MessageLogger, get_env_info, get_root_logger,
                         get_time_str, make_exp_dirs)
from basicsr.utils.options import dict2str, parse_options

def init_tb_loggers(opt):
    """Initialize tensorboard loggers.
    
    Args:
        opt (dict): Configuration. It contains:
            use_tb_logger (bool): Whether to use tensorboard logger
            tb_logger_path (str): Path to tensorboard logger
    """
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        from torch.utils.tensorboard import SummaryWriter
        tb_logger = SummaryWriter(log_dir=osp.join('experiments', opt['name']))
    else:
        tb_logger = None
    return tb_logger

def create_train_val_dataloader(opt, logger):
    """Create train and validation dataloaders.
    
    Args:
        opt (dict): Configuration. It contains:
            num_gpu (int): Number of GPUs
            dataset (dict): Dataset configuration
            num_workers (int): Number of workers
            batch_size (int): Batch size
        logger (logging.Logger): Logger
        
    Returns:
        tuple: Train dataloader, validation dataloader, train sampler
    """
    # Create train dataloader
    train_dataset = build_dataset(opt['datasets']['train'])
    train_sampler = EnlargedSampler(train_dataset, opt['world_size'], 
                                   opt['rank'], opt['batch_size'])
    train_loader = build_dataloader(
        train_dataset,
        opt['datasets']['train'],
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=train_sampler,
        seed=opt['manual_seed'])

    # Create validation dataloader
    val_dataset = build_dataset(opt['datasets']['val'])
    val_loader = build_dataloader(
        val_dataset,
        opt['datasets']['val'],
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=None,
        seed=opt['manual_seed'])

    return train_loader, val_loader, train_sampler

def train_pipeline(root_path):
    """Training pipeline.
    
    Args:
        root_path (str): Root path to configuration file
    """
    # Parse options
    opt = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    # Create experiment directories
    make_exp_dirs(opt)

    # Initialize loggers
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_loggers(opt)

    # Create dataloaders
    train_loader, val_loader, train_sampler = create_train_val_dataloader(opt, logger)

    # Create model
    model = build_model(opt)
    if opt['resume_state']:
        model.resume_training(opt['resume_state'])

    # Create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, model.start_iter, tb_logger)

    # Training
    logger.info(f'Start training from iteration: {model.start_iter}')
    data_timer, iter_timer = time.time(), time.time()
    start_time = time.time()

    # Create prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                        "Supported ones are: None, 'cuda', 'cpu'.")

    # Training loop
    for epoch in range(model.start_iter, opt['train']['total_iter']):
        # Reset training data loader
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer = time.time() - data_timer

            # Update learning rate
            model.update_learning_rate(epoch, train_data['lq'].shape[0])

            # Training
            model.feed_data(train_data)
            model.optimize_parameters(epoch)
            iter_timer = time.time() - iter_timer
            
            # Log and visualization
            if epoch % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter_timer': iter_timer, 'data_timer': data_timer}
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # Save models and validation
            if epoch % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch)
                model.save_training_state(epoch)

            # Validation
            if epoch % opt['val']['val_freq'] == 0:
                model.validation(val_loader, epoch, tb_logger)

            data_timer = time.time()
            iter_timer = time.time()
            train_data = prefetcher.next()

    # End training
    logger.info('End of training.')
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'Consumed total time: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1)  # -1 stands for the latest
    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    root_path = './configs/texture_enhancement.yml'
    train_pipeline(root_path)