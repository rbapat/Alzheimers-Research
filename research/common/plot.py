import os

import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


METRICS = [
    (0, "Average Loss"),
    (1, "Balanced Accuracy"),
    #     (2, "Sensitivity"),
    #     (3, "Specificity"),
]

PHASESET_1 = [
    (0, "train"),
    (1, "validation"),
    (2, "test"),
]

PHASESET_2 = [
    (0, "train"),
    (1, "test"),
]


def load_data(data_path, epoch_limit=0):
    data = torch.load(os.path.join(data_path))
    epoch_limit = data.shape[-2] if epoch_limit == 0 else epoch_limit

    return data[..., :epoch_limit, :]


def random_color():
    cmap = cm.get_cmap("tab10")
    random_index = np.random.randint(0, cmap.N)
    return cmap(random_index)


def plot_basic_split(result_path, save_path=None):
    results = load_data(os.path.join(result_path, "train_test.pt"))

    phaseset_1 = PHASESET_1[:2]
    colors = ["g", "r"]
    for metric_idx, metrix_name in METRICS:
        for phase_idx, phase_name in enumerate(phaseset_1):
            if phase_idx == 1:
                res = torch.mean(results[1:, :, metric_idx], dim=0)
                phase_name = (1, "test")
            else:
                res = results[phase_idx, :, metric_idx]
            print(res.shape)
            plt.plot(res, c=colors[phase_idx], label=phase_name[1])

        plt.title(f"{metrix_name} per epoch, 2 class")
        plt.xlabel("Epoch")
        plt.ylabel(metrix_name)
        plt.legend()

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"{metrix_name}.svg"))
        plt.figure()

    # plt.show()


def plot_nested_cv(
    experiments, plot_folds=False, plot_averages=False, epoch_limit=1000, save_path=None
):
    if plot_folds:
        for exp_name, exp_path in experiments:
            test_results = torch.load(os.path.join(exp_path, "test.pt"))

            for metric_idx, metric_name in METRICS:
                for phase_idx, phase_name in enumerate(PHASESET_2):
                    for fold_idx, fold_results in enumerate(test_results):
                        plt.plot(fold_results[phase_idx, :, metric_idx], label=fold_idx)

                    plt.title(f"{exp_name} | {phase_name}")
                    plt.xlabel("Epoch")
                    plt.ylabel(metric_name)
                    plt.legend()

                    if save_path is not None:
                        os.makedirs(save_path, exist_ok=True)
                        plt.savefig(
                            os.path.join(save_path, f"{exp_name}_{phase_name}.svg")
                        )
                    plt.figure()

    if plot_averages:
        test_results = {
            exp_name: (
                load_data(os.path.join(exp_path, "train.pt"), epoch_limit=epoch_limit),
                load_data(os.path.join(exp_path, "test.pt"), epoch_limit=epoch_limit),
            )
            for exp_name, exp_path in experiments
        }

        for metric_idx, metric_name in METRICS:
            for exp_name in test_results:
                plt.figure()
                train, test = test_results[exp_name]
                test = test[:, 1, :, metric_idx]
                train = train[:, :, 0, :, metric_idx]

                mean, std = torch.mean(train, dim=(0, 1)), torch.std(train, dim=(0, 1))
                x = torch.linspace(0, len(mean), len(mean))

                color = "g"  # random_color()
                plt.plot(mean, label="train", color=color)
                plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)

                mean, std = torch.mean(test, dim=0), torch.std(test, dim=0)
                x = torch.linspace(0, len(mean), len(mean))

                color = "r"  # random_color()
                plt.plot(mean, label="test", color=color)
                plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)

                plt.title(f"{metric_name} for {exp_name}")
                plt.xlabel("Epoch")
                plt.ylabel(metric_name)
                plt.legend(loc="lower right")
                plt.ylim((0, 1))

                if save_path is not None:
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(
                        os.path.join(save_path, f"{metric_name}_{exp_name}.svg")
                    )


def get_nestedcv_metrics(experiments):
    for exp_name, exp_path in experiments:
        # [5, 3, 2, 1000, 4], [5, 2, 1000, 4]
        train_results = torch.load(os.path.join(exp_path, "train.pt"))
        test_results = torch.load(os.path.join(exp_path, "test.pt"))

        outer, inner, phase, epoch, _ = train_results.shape
        train_results = train_results.reshape(outer * inner, phase, epoch, -1)

        for metric_idx, metric_name in METRICS:
            op = torch.min if metric_name == "Average Loss" else torch.max
            val_data = op(train_results[:, 1, :, metric_idx], dim=1).values
            test_data = op(test_results[:, 1, :, metric_idx], dim=1).values

            vmean = round(val_data.mean().item(), 4)
            vstd = round(val_data.std().item(), 4)
            tmean = round(test_data.mean().item(), 4)
            tstd = round(test_data.std().item(), 4)

            print(f"{exp_name} - {metric_name}")
            print(f"\tValidation {vmean} +/- {vstd}")
            print(f"\tTest {tmean} +/- {tstd}")


def main():
    np.random.seed(0)

    experiments = [
        (
            "MultiModalNet",
            "/home/rohan/projects/Alzheimers-Research/saved_results/best_prediction_MultiModalNet",
        ),
        # (
        #     "CVOnly",
        #     "/home/rohan/projects/Alzheimers-Research/saved_results/best_prediction_CVOnly",
        # ),
        # (
        #     "ImageOnly",
        #     "/home/rohan/projects/Alzheimers-Research/saved_results/best_prediction_ImageOnly",
        # ),
    ]

    # PREDICTION_SAVE_PATH = "/home/rohan/projects/Alzheimers-Research/research/figures/vectorimgs/best_prediction_MultiModalNet"
    # get_nestedcv_metrics(experiments)
    # plot_nested_cv(experiments, plot_averages=True, save_path=PREDICTION_SAVE_PATH)
    # plt.show()

    CLASSIFICATION_SAVE_PATH = "/home/rohan/projects/Alzheimers-Research/research/figures/vectorimgs/best_classification_DenseNet"
    plot_basic_split(
        "/home/rohan/projects/Alzheimers-Research/saved_results/best_classification_DenseNet",
        save_path=CLASSIFICATION_SAVE_PATH,
    )


if __name__ == "__main__":
    main()
