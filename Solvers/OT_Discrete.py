from tqdm import tqdm
import torch
import numpy as np
from Models.ot_models import PotentialMLP
from Solvers.DefenseTrain import Defense_Train_Base
from Utils import utils


class OT_Discrete(Defense_Train_Base):
    
    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name = "OT_Discrete")

    def train(self, dataloader_train, dataloader_test):
        self.OT_D_train(dataloader_train, None)
        self.OT_D_test(None, dataloader_test)

    def OT_D_train(self, dataloader_train, dataloader_valid, stage = "OT_D-train", flad_load_ckp = True):

        model = PotentialMLP(dim_in = (self.cfg_m.img_size**2)*2, dim_out = self.cfg_m.img_size**2, hidden_num = self.cfg_m.MLP_hidden_num).to(self.device)
        loss_func = dual_obj_loss(img_size = self.cfg_m.img_size, epsilon = self.cfg_m.epsilon, device = self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr = self.cfg_m.learning_rate_init)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = self.cfg_m.epochs)
        if flad_load_ckp and self.cfg_proj.flag_load is not None:
            model, optimizer, lr_scheduler, epoch_start = self.load_ckp(model, optimizer, lr_scheduler, stage)
        else:
            epoch_start = 0

        if epoch_start < self.cfg_m.epochs:
            pbar = tqdm(initial=epoch_start, total = self.cfg_m.epochs)
            pbar.set_description("%s_%s, epoch = %d, loss = None"%(self.name, stage, 0))
            for epoch in range(epoch_start, self.cfg_m.epochs, 1):
                loss_trace = []
                for batch_idx, (_, _, x_a, x_b) in enumerate(dataloader_train):
                
                    x_a = x_a.to(self.device)
                    x_b = x_b.to(self.device)

                    f_pred = model(x_a, x_b) 
                    loss = loss_func(x_a, x_b, f_pred)  
                    loss_trace.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()  
                    optimizer.step()        
                    str_record = "%s_%s, epoch = %d, loss_batch = %.8f"%(self.name, stage, epoch, loss.item())
                    pbar.set_description(str_record)    
                lr_scheduler.step()
                #validation
                loss_epoch_avg = np.mean(loss_trace)
                pbar.update(1)
                str_record = "%s_%s, epoch = %d, loss = %.8f"%(self.name, stage, epoch, loss_epoch_avg)
                pbar.set_description(str_record)
                self.logger.info(str_record)
                if (epoch+1) >= self.cfg_m.log_interval and (epoch+1) % self.cfg_m.log_interval == 0 or (epoch+1) == self.cfg_m.epochs:
                    self.save_ckp(model, optimizer, lr_scheduler, (epoch+1), stage)
            pbar.close()


    def OT_D_test(self, dataloader_train, dataloader_valid):
        model = PotentialMLP(dim_in = self.cfg_m.img_size**2*2, dim_out = self.cfg_m.img_size**2, hidden_num = self.cfg_m.MLP_hidden_num).to(self.device)
        model, _, _, _ = self.load_ckp(model, None, None, "OT_D-train")
        loss_func = dual_obj_loss(img_size = self.cfg_m.img_size, epsilon = self.cfg_m.epsilon, device = self.device)
        model.eval()
        #pred
        for batch_idx, (_, _, xs_a, xs_b) in enumerate(dataloader_valid):
            xs_a = xs_a[:2].to(self.device)
            xs_b = xs_b[:2].to(self.device)
            break
        Ps_gt = Cal_P(xs_a, xs_b, True, loss_func, model)
        Ps_pred = Cal_P(xs_a, xs_b, False, loss_func, model)
        imgs_gt = interp(Ps_gt[0], num_inter = 11, batch_size = 50000, img_size = self.cfg_m.img_size)
        imgs_pred = interp(Ps_pred[0], num_inter = 11, batch_size = 50000, img_size = self.cfg_m.img_size)
        utils.save_r(imgs_gt, xs_a[0], xs_b[0], path = self.log_sub_folder, title = "GroundTruth")
        utils.save_r(imgs_pred, xs_a[0], xs_b[0], path = self.log_sub_folder, title = "Pred")


def Cal_P(x_a, x_b, flag_ground_truth, loss_func = None, model = None):
    if flag_ground_truth:
        from ott.geometry.pointcloud import PointCloud
        import jax.numpy as jnp
        from ott.core import quad_problems, problems, sinkhorn
        from jax.config import config
        config.update("jax_enable_x64", True)
        x_grid = []
        for i in jnp.linspace(1, 0, num = loss_func.img_size):
            for j in jnp.linspace(0, 1, num = loss_func.img_size):
                x_grid.append([j, i])
        x_grid = jnp.array(x_grid)
        geom = PointCloud(x = x_grid, y = x_grid, epsilon = loss_func.epsilon, online=True)
        x_a_jnp = jnp.array(x_a.data.cpu().numpy())
        x_b_jnp = jnp.array(x_b.data.cpu().numpy())
        P = []
        for a, b in zip(x_a_jnp, x_b_jnp):
            ot_prob = problems.LinearProblem(geom, a = a, b = b)
            solver = sinkhorn.make(lse_mode=True)
            out_p = solver(ot_prob)
            P.append(np.array(out_p.matrix))
        P = np.array(P)
    else:
        f_pred = model(x_a, x_b) 
        P = loss_func.pred_transport(x_a, x_b, f_pred)
    return P


class cost_matrix_calculator():
    def __init__(self, img_size, epsilon):
        self.img_size = img_size
        x_grid = []
        for i in np.linspace(1, 0, num = img_size):
            for j in np.linspace(0, 1, num = img_size):
                x_grid.append([j, i])
        x_grid = np.array(x_grid, dtype = np.float64)
        self.x = x_grid
        self.y = x_grid
        self.epsilon = epsilon
        self.power = 2.0

    def compute_C_K(self):
        C = -2*self.x@self.y.T
        C += self.norm(self.x)[:, np.newaxis] + self.norm(self.y)[np.newaxis, :]
        K = np.exp(-C/self.epsilon)
        return C, K

    def norm(self, x):
        return np.sum(x ** 2, axis=-1)


class dual_obj_loss():
    def __init__(self, img_size, epsilon, device):
        self.img_size = img_size
        self.device = device
        self.epsilon = epsilon
        c_cal = cost_matrix_calculator(img_size, epsilon)
        cost_matrix, kernel_matrix = c_cal.compute_C_K()
        self.K = torch.tensor(kernel_matrix, dtype = torch.float64).to(device)
        self.C = torch.tensor(cost_matrix, dtype = torch.float64).to(device)

    def g_from_f(self, a, b, f):
        g_sink = self.epsilon*torch.log(b) - self.epsilon*torch.log(torch.exp(f/self.epsilon)@(self.K))
        f_sink = self.epsilon*torch.log(a) - self.epsilon*torch.log(torch.exp(g_sink/self.epsilon)@((self.K).T))
        return g_sink, f_sink

    def dual_obj_from_f(self, a, b, f):
        g_sink, f_sink = self.g_from_f(a, b, f)
        g_sink_nan = torch.nan_to_num(g_sink, nan=0.0, posinf=0.0, neginf=0.0)
        f_sink_nan = torch.nan_to_num(f_sink, nan=0.0, posinf=0.0, neginf=0.0)
        dual_obj_left = torch.sum(f_sink_nan * a, dim=-1) + torch.sum(g_sink_nan * b, dim=-1)
        dual_obj_right = - self.epsilon*torch.sum(torch.exp(f_sink/self.epsilon)*(torch.exp(g_sink/self.epsilon)@(self.K)), dim = -1)
        dual_obj = dual_obj_left + dual_obj_right
        return dual_obj

    def pred_transport(self, a, b, f_pred):
        g_sink, f_sink = self.g_from_f(a, b, f_pred)
        f_expand = f_sink.unsqueeze(2)
        g_expand = g_sink.unsqueeze(1)
        P = torch.matmul(torch.exp(f_expand/self.epsilon), torch.exp(g_expand/self.epsilon))*self.K
        P = P.data.cpu().numpy()
        return P

    def __call__(self, a, b, f_pred):
        dual_value = self.dual_obj_from_f(a, b, f_pred)
        loss =  - torch.mean(dual_value)
        return loss


def interp(P, num_inter, batch_size, img_size):
    P_flatten = P.flatten()
    grid = []
    for i in np.linspace(1, 0, num = img_size):
        for j in np.linspace(0, 1, num = img_size):
            grid.append([j, i])
    x_grid = np.array(grid)
    y_grid = np.array(grid)

    def get_hist(t, P_flat):
        map_samples = np.random.choice(range(len(P_flat)), size = batch_size, p = P_flat)
        a_samples = x_grid[map_samples // img_size**2]
        b_samples = y_grid[map_samples % img_size**2]
        proj_samples = (1.-t)*a_samples + t*b_samples
        hist, _, _ = np.histogram2d(proj_samples[:,1], proj_samples[:,0], bins = np.linspace(0., 1., num = img_size + 1))

        hist = np.flipud(hist)
        thresh = np.quantile(hist, 0.9)
        hist[hist > thresh] = thresh
        hist = hist / hist.max()
        return hist

    hists = []
    ts = np.linspace(0, 1, num = num_inter)

    for i, t in enumerate(ts):
        hist = get_hist(t, P_flatten)
        hists.append(hist)
    return hists
