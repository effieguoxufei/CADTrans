import os
import torch
import argparse
from tqdm import tqdm
from config import *
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataset import SolidData, ProfileData, LoopData
from model.encoder import SolidEncoder, ProfileEncoder, LoopEncoder
from model.decoder import SolidDecoder, ProfileDecoder, LoopDecoder
from model.discriminator import SolidDiscriminator, ProfileDiscriminator, LoopDiscriminator
from model.network import get_constant_schedule_with_warmup, squared_emd_loss


def parse_aug(format):
    """
    Find the corresponding function to run
    """
    data_func = {
        "solid": SolidData,
        "profile": ProfileData,
        "loop": LoopData
    }[format]

    data_path = {
        "solid": SOLID_TRAIN_PATH,
        "profile": PROFILE_TRAIN_PATH,
        "loop": LOOP_TRAIN_PATH
    }[format]

    enc_func = {
        "solid": SolidEncoder,
        "profile": ProfileEncoder,
        "loop": LoopEncoder
    }[format]

    dec_func = {
        "solid": SolidDecoder,
        "profile": ProfileDecoder,
        "loop": LoopDecoder
    }[format]
    
    dis_func = {
        "solid": SolidDiscriminator,
        "profile": ProfileDiscriminator,
        "loop": LoopDiscriminator
    }[format]

    return data_func, data_path, enc_func, dec_func, dis_func


def train(args):
    # gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    data_func, data_path, enc_func, dec_func,dis_func = parse_aug(args.format)
    
    # Initialize dataset loader
    dataset = data_func(data_path)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=True, 
                                             batch_size=args.batchsize,
                                             num_workers=6)
    
    # Initialize models
    encoder = enc_func()
    encoder = encoder.cuda().train()

    decoder = dec_func()
    decoder = decoder.cuda().train()

    disc = dis_func()
    disc = disc.cuda().train()



    params_vq = list(decoder.parameters()) + list(encoder.parameters())
    optimizer_vq = torch.optim.AdamW(params_vq, 
        lr=1e-4, eps=1e-08,betas=(args.beta1, args.beta2)
        )
    scheduler_vq = get_constant_schedule_with_warmup(optimizer_vq, 2000)

    params_gan = list(disc.parameters())
    optimizer_gan = torch.optim.AdamW(params_gan, 
        lr=1e-4, eps=1e-08,betas=(args.beta1, args.beta2)
        )
    scheduler_gan = get_constant_schedule_with_warmup(optimizer_gan, 2000)
    writer = SummaryWriter(log_dir=args.output)
    
    # Main training loop
    iters = 0
    step_per_epoch=len(dataloader)
    print('Start training...')
    for epoch in range(TOTAL_TRAIN_EPOCH):  
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        # for param, seq_mask, ignore_mask,_ in dataloader:
        for i, (param, seq_mask, ignore_mask, _) in enumerate(dataloader):
            param = param.cuda()
            seq_mask = seq_mask.cuda()
            ignore_mask = ignore_mask.cuda()
           
            # Pass through encoder 
            latent_code, vq_loss, selection, _ = encoder(param, seq_mask)
            
            # Pass through decoder
            param_logits = decoder(param, seq_mask, ignore_mask, latent_code)


            param_real=F.one_hot(param, num_classes=param_logits.shape[-1]).float()
            disc_real = disc(param_real)
            param_fake=param_logits

            disc_fake = disc(param_fake)


            disc_factor=adopt_weight(args.disc_factor,epoch*step_per_epoch+i,threshold=args.disc_start)
            
            # Compute loss                                    
            param_loss = squared_emd_loss(logits=param_logits, 
                                            labels=param, 
                                            num_classes=param_logits.shape[-1], 
                                            mask=ignore_mask)


            g_loss=-torch.mean(disc_fake)
            last_layer_weight=decoder.getweight()
            λ=calculate_lambda(param_loss,g_loss,last_layer_weight)

            codebook_loss=param_loss + vq_loss+disc_factor*λ*g_loss

            d_loss_real=torch.mean(F.relu(1.-disc_real))
            d_loss_fake=torch.mean(F.relu(1.+disc_fake))

            gan_loss=disc_factor*0.5*(d_loss_real+d_loss_fake)



            # logging
            if iters % 100 == 0:
                writer.add_scalar("Loss/codebook", codebook_loss, iters) 
                writer.add_scalar("Loss/gan",gan_loss,iters)
                writer.add_scalar("Loss/param", param_loss, iters) 
                writer.add_scalar("Loss/vq", vq_loss, iters) 


            # Update model
            optimizer_vq.zero_grad()
            codebook_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(params_vq, max_norm=1.0)  # clip gradient
            optimizer_gan.zero_grad()
            gan_loss.backward()
            nn.utils.clip_grad_norm_(params_gan, max_norm=1.0)  # clip gradient
            optimizer_vq.step()
            scheduler_vq.step()  # linear warm up to 1e-3
            optimizer_gan.step()
            scheduler_gan.step()  # linear warm up to 1e-3
            iters += 1
            progress_bar.update(1)

        progress_bar.close()
        print("Loss{}: gan {:.4f}     codebook {:.4f}  ".format(epoch,gan_loss.item(), codebook_loss.item()))
        writer.flush()

        # Re-init codebook 
        if epoch<REINIT_TRAIN_EPOCH:
            # Compute cluster data count & data to cluster distance
            code_encoded = []
            for param, seq_mask, _,_ in dataloader:
                param = param.cuda()
                seq_mask = seq_mask.cuda()
                with torch.no_grad():
                    _, _code_encoded_ = encoder.count_code(param, seq_mask)
                    code_encoded.append(_code_encoded_.reshape(-1,256).detach().cpu())
          
            code_encoded = torch.vstack(code_encoded)
            code_encoded = code_encoded[torch.randperm(code_encoded.size()[0])] # random shuffle
            reinit_count = encoder.codebook.reinit(code_encoded)

        # Save model after n epoch
        if (epoch+1) % 50 == 0:
            torch.save(encoder.state_dict(), os.path.join(args.output,'enc_epoch_'+str(epoch+1)+'.pt'))
            torch.save(decoder.state_dict(), os.path.join(args.output,'dec_epoch_'+str(epoch+1)+'.pt'))
            torch.save(decoder.state_dict(), os.path.join(args.output,'disc_epoch_'+str(epoch+1)+'.pt'))
       
    writer.close()


def adopt_weight(disc_factor,i,threshold,value=0.):
    if i<threshold:
        disc_factor=value
    return disc_factor

def calculate_lambda(perceptual_loss,gan_loss,last_layer_weight):
        perceptual_loss_grads=torch.autograd.grad(perceptual_loss,last_layer_weight,retain_graph=True)[0]
        gan_loss_gards=torch.autograd.grad(gan_loss,last_layer_weight,retain_graph=True)[0]

        λ=torch.norm(perceptual_loss_grads)/(torch.norm(gan_loss_gards)+1e-4)
        λ=torch.clamp(λ,0,1e4).detach()
        return 0.8*λ


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output folder to save the data", required=True)
    parser.add_argument("--batchsize", type=int, help="Batch size", required=True)
    parser.add_argument("--device", type=str, help="CUDA device", required=True)
    parser.add_argument("--format", type=str, help="Data type", required=True)
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
  
    args = parser.parse_args()

    # Create training folder
    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    # Start training 
    train(args)
