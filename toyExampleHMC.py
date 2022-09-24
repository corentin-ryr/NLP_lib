import torch
import matplotlib.pyplot as plt

import torch
import torch.distributions
import numpy as np

from Mixers.Datasets.DSP import  ToyDataset
from Mixers.Models.toyModel import ToyModel
from Mixers.Trainers.trainerDirector import TrainerDirector




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f"Working on {device}")


def model_loss(y_preds, y):
    (mean, var) = y_preds
    delta = (mean - y)
    loss = 0.5 * delta * delta / var + 0.5*torch.log(var)
    print(loss.shape)
    return loss.sum(0)/y.shape[0] + 0.5 * np.log(2 * np.pi)


## Get the dataset ==================================================
traindataset = ToyDataset()
testdataset = ToyDataset(train=False)


## Set up of the hyperparameters =====================================
num_samples = 100
sigma_prior = 0.1
step_size = 3e-4
burn_in = (int)(0. * num_samples) #Arbitrary, not from the paper

trajectory_length = 3.14 * sigma_prior / 2
num_steps_per_sample = (int)(trajectory_length // step_size)


## Sample the first chain ============================================
model = ToyModel(sigma=0.005).to(device)

trainer = TrainerDirector.get_hmc_trainer(model=model, traindataset=traindataset, testdataset=testdataset, device=device, batch_size=256) 

trainer.summarize_model()

trainer.train()


# trainer = Trainer(model, sigma_p=sigma_prior)
# params_hmc = trainer.hmc((x_train_inputs, y_train_inputs), num_samples, step_size, num_steps_per_sample, burn_in=burn_in, device=device)
# params_hmc = params_hmc[-70:]

# pred_list, log_prob_list = hamiltorch.predict_model(model, params_hmc, x=x_truth_inputs, y=y_truth_inputs, model_loss='regression')

# pred_mean_1 = torch.mean(pred_list, dim=0)[:, 0].cpu()
# pred_std_1 = torch.std(pred_list, dim=0)[:, 0].cpu()
# print(torch.max(pred_std_1))


# nll = model_loss((pred_mean_1, pred_std_1), torch.tensor(y_truth[:, 0]))
# print(nll)
# print(-float(nll.detach()))

# # params_hmc = torch.tensor(np.array(params_hmc))
# torch.save(params_hmc, "Results/HMC/HMC_parameters")

# ## Sample the second chain ===========================================
# model = ToyModel(sigma=0.005).to(device)
# trainer = Trainer(model, sigma_p=sigma_prior)
# params_hmc = trainer.hmc((x_train_inputs, y_train_inputs), num_samples, step_size, num_steps_per_sample, burn_in=burn_in, device=device)
# params_hmc = params_hmc[-70:]

# pred_list, log_prob_list = hamiltorch.predict_model(model, params_hmc, x=x_truth_inputs, y=y_truth_inputs, model_loss='regression')

# pred_mean_2 = torch.mean(pred_list, dim=0)[:, 0].cpu()
# pred_std_2 = torch.std(pred_list, dim=0)[:, 0].cpu()


# fig, ax = plt.subplots(1, 3)
# ax[0].set_title("(a) Chain 1")
# ax[0].plot(x_truth, y_truth, color='black')
# ax[0].plot(x_train, y_train, 'o', color='red')
# ax[0].fill_between(x_truth, pred_mean_1 - pred_std_1, pred_mean_1 + pred_std_1, alpha=0.5, color='#FFBB13')
# ax[0].plot(x_truth, pred_mean_1, color='g')
# ax[0].plot(x_truth, pred_mean_1 + pred_std_1, color='#FF7114')
# ax[0].plot(x_truth, pred_mean_1 - pred_std_1, color='#FF7114')
# ax[0].set_ylim(-5, 5)

# ax[1].set_title("(b) Chain 2")
# ax[1].plot(x_truth, y_truth, color='black')
# ax[1].plot(x_train, y_train, 'o', color='red')
# ax[1].fill_between(x_truth, pred_mean_2 - pred_std_2, pred_mean_2 + pred_std_2, alpha=0.5, color='#FF3A13')
# ax[1].plot(x_truth, pred_mean_2, color='g')
# ax[1].plot(x_truth, pred_mean_2 + pred_std_2, color='#FF320A')
# ax[1].plot(x_truth, pred_mean_2 - pred_std_2, color='#FF320A')
# ax[1].set_ylim(-5, 5)

# ax[2].set_title("(c) Overlaid")
# ax[2].plot(x_truth, y_truth, color='black')
# ax[2].plot(x_train, y_train, 'o', color='red')
# ax[2].fill_between(x_truth, pred_mean_1 - pred_std_1, pred_mean_1 + pred_std_1, alpha=0.5, color='#FFBB13')
# ax[2].plot(x_truth, pred_mean_1, color='g')
# ax[2].plot(x_truth, pred_mean_1 + pred_std_1, color='#FF7114')
# ax[2].plot(x_truth, pred_mean_1 - pred_std_1, color='#FF7114')
# ax[2].fill_between(x_truth, pred_mean_2 - pred_std_2, pred_mean_2 + pred_std_2, alpha=0.5, color='#FF3A13')
# ax[2].plot(x_truth, pred_mean_2, color='g')
# ax[2].plot(x_truth, pred_mean_2 + pred_std_2, color='#FF320A')
# ax[2].plot(x_truth, pred_mean_2 - pred_std_2, color='#FF320A')
# ax[2].set_ylim(-5, 5)


# fig.savefig("Result.png")
# plt.show()
