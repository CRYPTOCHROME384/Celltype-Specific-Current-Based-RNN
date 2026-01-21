import matplotlib.pyplot as plt
import torch as tch
import numpy as np

def plot_loss(epoch, losses, title, tag = ''):
    plt.figure()
    plt.semilogy(range(epoch), losses[:epoch])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(tag + '.png',  bbox_inches = 'tight')
    plt.close()


def plot_neurons(net, rates, default_parameters, t_del = -.03, tag = ''):    
    with tch.set_grad_enabled(False):
        y_alm_l = rates['ALM_l']
        y_alm_r = rates['ALM_r']
        input_net = {'ALM_l':y_alm_l}
        results, _ = net(input_net)
        rates_alm_l = results['rates_alm_l']
        rates_alm_r = results['rates_alm_r']
        rates_octx_l = results['rates_octx_l']
        rates_octx_r = results['rates_octx_r']
        rates_alm_l = rates_alm_l.detach().cpu().numpy()
        rates_alm_r = rates_alm_r.detach().cpu().numpy()
        rates_octx_l = rates_octx_l.detach().cpu().numpy()
        rates_octx_r = rates_octx_r.detach().cpu().numpy()
        y_alm_l = y_alm_l.detach().cpu().numpy()
        y_alm_r = y_alm_r.detach().cpu().numpy()
        t_start = default_parameters['t_start']
        dt = default_parameters['dt']
        n_time = y_alm_l.shape[1]
        x = np.linspace(0, n_time, n_time) * dt
        x = x + t_start
        n_neurons = 1000
        for j in range(0, n_neurons,10):
            plt.figure(figsize=(6, 3))
            plt.axvline(x=net.t_min_input + t_del, color='gray', linestyle='--')
            plt.axvline(x=net.t_go + t_del, color='gray', linestyle='--') 
            plt.axvline(x=net.t_max_input + t_del, color='gray', linestyle='--')
            plt.plot(x, rates_alm_l[0, :, j], 'r--',  lw =3) 
            plt.plot(x, y_alm_l[0, :, j], 'r-', lw =3) 
            plt.plot(x, rates_alm_l[1, :, j], 'b--',  lw =3) 
            plt.plot(x, y_alm_l[1, :, j], 'b-', lw =3) 
            plt.xlabel('Time (s)')
            plt.ylabel('Mode value')
            plt.title("ALM L")
            plt.savefig('out/neuronal_dynamics/neural_ALM_L_'+ tag + '{}.png'.format(j),  bbox_inches = 'tight')
            plt.close()
        for j in range(0, n_neurons,10):
            plt.figure(figsize=(6, 3))
            plt.axvline(x=net.t_min_input + t_del, color='gray', linestyle='--')
            plt.axvline(x=net.t_go + t_del, color='gray', linestyle='--') 
            plt.axvline(x=net.t_max_input + t_del, color='gray', linestyle='--')
            plt.plot(x, rates_alm_r[0, :, j], 'r--',  lw =3) 
            plt.plot(x, y_alm_r[0, :, j], 'r-', lw =3) 
            plt.plot(x, rates_alm_r[1, :, j], 'b--',  lw =3) 
            plt.plot(x, y_alm_r[1, :, j], 'b-', lw =3) 
            plt.xlabel('Time (s)')
            plt.ylabel('Mode value')
            plt.title('ALM R')
            plt.savefig('out/neuronal_dynamics/neural_ALM_R_'+ tag + '{}.png'.format(j),  bbox_inches = 'tight')
            plt.close()
        for j in range(0, 200,20):
            plt.figure(figsize=(6, 3))
            plt.axvline(x=net.t_min_input + t_del, color='gray', linestyle='--')
            plt.axvline(x=net.t_go + t_del, color='gray', linestyle='--') 
            plt.axvline(x=net.t_max_input + t_del, color='gray', linestyle='--')
            plt.plot(x, rates_octx_l[0, :, j], 'r-',  lw =3) 
            plt.plot(x, rates_octx_l[1, :, j], 'b-',  lw =3) 
            plt.xlabel('Time (s)')
            plt.ylabel('Mode value')
            plt.title("ALM L")
            plt.savefig('out/neuronal_dynamics/neural_octx_L_'+ tag + '{}.png'.format(j),  bbox_inches = 'tight')
            plt.close()
        for j in range(0, 200,20):
            plt.figure(figsize=(6, 3))
            plt.axvline(x=net.t_min_input + t_del, color='gray', linestyle='--')
            plt.axvline(x=net.t_go + t_del, color='gray', linestyle='--') 
            plt.axvline(x=net.t_max_input + t_del, color='gray', linestyle='--')
            plt.plot(x, rates_octx_r[0, :, j], 'r-',  lw =3) 
            plt.plot(x, rates_octx_r[1, :, j], 'b-',  lw =3) 
            plt.xlabel('Time (s)')
            plt.ylabel('Mode value')
            plt.title('ALM R')
            plt.savefig('out/neuronal_dynamics/neural_octx_R_'+ tag + '{}.png'.format(j),  bbox_inches = 'tight')
            plt.close()


def plot_latents(net,  rates, default_parameters, t_del = -.03, tag = ''):   
    with tch.set_grad_enabled(False):
        y_alm_l = rates['ALM_l']
        input_net = {'ALM_l':y_alm_l}
        results, _ = net(input_net)
        latents_alm_l = results['latents_alm_l']
        latents_alm_r = results['latents_alm_r']
        latents_alm_l = latents_alm_l.detach().cpu().numpy()
        latents_alm_r = latents_alm_r.detach().cpu().numpy()
     
   
        t_start = default_parameters['t_start']
        dt = default_parameters['dt']
        n_time = y_alm_l.shape[1]
        x = np.linspace(0, n_time, n_time) * dt
        x = x + t_start
        for j in range(latents_alm_l.shape[2]):
            plt.figure(figsize=(6, 3))
            plt.axvline(x=net.t_min_input + t_del, color='gray', linestyle='--')
            plt.axvline(x=net.t_go + t_del, color='gray', linestyle='--') 
            plt.axvline(x=net.t_max_input + t_del, color='gray', linestyle='--')
            plt.plot(x, latents_alm_l[0, :, j], 'r-',  lw =3) 
            plt.plot(x, latents_alm_l[1, :, j], 'b-',  lw =3) 
            plt.xlabel('Time (s)')
            plt.ylabel('Mode value')
            plt.title("ALM L")
            plt.savefig('out/latents/latents_ALM_L_'+tag + '{}.png'.format(j), bbox_inches = 'tight')
            plt.close()
        for j in range(latents_alm_r.shape[2]):
            plt.figure(figsize=(6, 3))
            plt.axvline(x=net.t_min_input + t_del, color='gray', linestyle='--')
            plt.axvline(x=net.t_go + t_del, color='gray', linestyle='--') 
            plt.axvline(x=net.t_max_input + t_del, color='gray', linestyle='--')
            plt.plot(x, latents_alm_r[0, :, j], 'r-',  lw =3) 
            plt.plot(x, latents_alm_r[1, :, j], 'b-',  lw =3) 
            plt.xlabel('Time (s)')
            plt.ylabel('Mode value')
            plt.title("ALM R")
            plt.savefig('out/latents/latents_ALM_R_'+tag + '{}.png'.format(j), bbox_inches = 'tight')
            plt.close()


