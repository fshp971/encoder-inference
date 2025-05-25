import argparse, numpy as np, os
import torch, torchvision


def main(args):
    sz = 224
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((sz, sz)),
        torchvision.transforms.ToTensor(),])

    dataset = torchvision.datasets.ImageFolder(
        args.data_folder_path, transform=trans)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=4)

    num = len(dataset)
    np_data = np.zeros((num, sz, sz, 3), dtype=np.uint8)
    np_target = np.zeros(num, dtype=np.int64)

    st = 0
    for x, y in loader:
        ed = st + len(x)
        xx = (x * 255).round().clamp(0, 255).type(torch.uint8).permute((0, 2, 3, 1)).numpy()
        yy = y.numpy()
        np_data[st : ed] = xx
        np_target[st : ed] = yy
        st = ed

    save_path = os.path.join(args.save_path, "imagenet-val.npz")
    np.savez(save_path, data=np_data, target=np_target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    main(args)
