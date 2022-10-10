from mixers.trainers.classificationTrainers import ClassificationTrainerHMC
from mixers.datasets.DSP import  ToyDataset
from mixers.models.toyModel import ToyModel
import mixers.trainers.hamiltorch as hamiltorch
from mixers.utils.helper import get_device

import torch
from torch.utils.data import DataLoader

from datetime import datetime
import matplotlib.pyplot as plt

useGPU = True

if __name__ == "__main__":
    device = get_device(useGPU)

    ## Get the dataset ==================================================
    traindataset = ToyDataset()
    testdataset = ToyDataset(train=False)

    x = []
    y = []
    x_truth = []
    y_truth = []
    for data in DataLoader(traindataset):
        x.append(data[0][0])
        y.append(data[1][0])
    for data in DataLoader(testdataset):
        x_truth.append(data[0][0])
        y_truth.append(data[1][0])

    x = torch.stack(x)
    y = torch.stack(y)
    x_truth = torch.stack(x_truth)
    y_truth = torch.stack(y_truth)

    ## Print the data =====================================
    # plt.title("Sampled points and the underlying distribution")
    # plt.plot(x, y, 'o', color='red')
    # plt.plot(x_truth, y_truth, color='black')
    # plt.show()


    ## Sample the first chain ============================================
    num_samples = 40
    model = ToyModel().to(device)

    # trainer = TrainerDirector.get_hmc_trainer(model=model, traindataset=traindataset, testdataset=testdataset, device=device, batch_size=256, num_samples=num_samples) 

    trainer = ClassificationTrainerHMC(model, device=device, traindataset=traindataset, testdataset=testdataset, batch_size=256, 
                                        num_samples=num_samples, num_classes=2)


    trainer.summarize_model()

    trainer.train()

    trainer.save_model()
    # trainer.load_model("saves/ToyModel-2022-09-25-20-14")

    params_hmc = trainer.params_hmc_f

    pred_list = hamiltorch.inference_model(model, params_hmc, x_truth)
    pred_list = pred_list[10:]

    print(len(pred_list))

    pred_mean_1 = torch.mean(pred_list, dim=0)[:, 0].cpu()
    pred_std_1 = torch.std(pred_list, dim=0)[:, 0].cpu()
    print(f"Maximum incertainty: {torch.max(pred_std_1)}")


   
    plt.title("(a) Chain 1")
    plt.plot(x_truth[:, 0], y_truth, color='black')
    plt.plot(x[:, 0], y, 'o', color='red')
    plt.fill_between(x_truth[:, 0], pred_mean_1 - pred_std_1, pred_mean_1 + pred_std_1, alpha=0.5, color='#FFBB13')
    plt.plot(x_truth[:, 0], pred_mean_1, color='g')
    plt.plot(x_truth[:, 0], pred_mean_1 + pred_std_1, color='#FF7114')
    plt.plot(x_truth[:, 0], pred_mean_1 - pred_std_1, color='#FF7114')

    plt.savefig("Result"  + "-" + datetime.today().strftime('%Y-%m-%d-%H-%M') + ".png")
    plt.show()
