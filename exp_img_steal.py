import yaml, argparse, os, torch, numpy as np
from PIL import Image
from io import BytesIO
import torchvision

import utils, enatk


class NumpyDatasetDualTrans():
    def __init__(self, root: str, trans1, trans2):
        dataset = np.load(root)
        self.data = dataset["data"]
        self.target = dataset["target"]
        self.trans1 = trans1
        self.trans2 = trans2

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        return self.trans1(img), self.trans2(img)

    def __len__(self):
        return len(self.data)


class SubDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class StealDataset:
    @torch.no_grad()
    @staticmethod
    def build(
        dataset: NumpyDatasetDualTrans,
        steal_sz: int,
        encoder,
        classifier,
        device: str,
        inferred_encoder=None,
        inferred_device: str = None,
        cache_batch: int = 1000,
    ) -> "StealDataset":

        indices = np.random.permutation(np.arange(len(dataset)))[:steal_sz]
        subset = SubDataset(dataset, indices)
        loader = utils.data.Loader(dataset=subset, batch_size=cache_batch, train=False)

        xx, yy = [], []

        for x, steal_x in loader:
            yy.append(classifier(encoder(x.to(device))).to("cpu"))
            if inferred_encoder is None:
                xx.append(steal_x)
            else:
                xx.append(inferred_encoder(steal_x.to(inferred_device)).to("cpu"))

        xx = torch.cat(xx)
        yy = torch.cat(yy)

        return StealDataset(xx, yy)

    def __init__(self, xx, yy):
        self.xx = xx
        self.yy = yy

    def __getitem__(self, idx):
        return self.xx[idx], self.yy[idx]

    def __len__(self):
        return len(self.xx)


def get_args():
    parser = argparse.ArgumentParser()

    utils.add_shared_args(parser)
    utils.add_img_steal_exp_args(parser)
    utils.add_cls_task_args(parser)

    return parser.parse_args()


@torch.no_grad()
def evaluate(embed_trainloader, embed_testloader,
               trainloader, testloader,
               classifier, steal_model, device, device_steal):

    steal_model.eval()
    cnt, acc, agree = 0, 0, 0
    for (embed_x, embed_y), (x, y) in zip(embed_testloader, testloader):
        assert len(embed_y) == len(y) and (y != embed_y).sum() == 0
        target_y = classifier(embed_x.to(device)).argmax(dim=1).to(device_steal)
        _y = steal_model(x.to(device_steal)).argmax(dim=1)

        cnt += len(y)
        acc += (y.to(device_steal) == _y).sum().item()
        agree += (target_y == _y).sum().item()

    acc /= cnt
    agree /= cnt

    return acc, agree


def exp_steal(args, cfg: dict, logger):
    device = "cuda:0"
    device_steal = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    encoder = utils.model.get_img_encoder(args.encoder, device=device)
    classifier = torch.load(args.classifier_path, map_location=device)

    # pei_assist = cfg.get("pei_assist", False)
    pei_assist = args.pei_assist

    if pei_assist:
        inferred_encoder = utils.model.get_img_encoder(args.inferred_encoder, device=device_steal)
    else:
        inferred_encoder = None

    steal_model_pms = cfg.get("steal_model_pms", {})
    if pei_assist:
        steal_model_pms.update({"in_dims": inferred_encoder.embed_dims, "out_dims": args.cls_num})
    else:
        steal_model_pms.update({"in_dims": 3, "out_dims": args.cls_num})
    steal_model = utils.model.get_classifier(cfg["steal_model"], **steal_model_pms)
    steal_model.to(device_steal)

    optim = utils.get_optim(cfg["optim"], steal_model.parameters(), **cfg["optim_pms"])
    scheduler = utils.get_scheduler(cfg["scheduler"], optim, **cfg["scheduler_pms"])

    target_trans = lambda x: encoder.trans(x)[0]
    if pei_assist:
        steal_trans = lambda x: inferred_encoder.trans(x)[0]
    else:
        steal_trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),])

    orig_steal_dataset = NumpyDatasetDualTrans(
        root=args.steal_npz_data_path, trans1=target_trans, trans2=steal_trans)

    logger.info("start building surrogate dataset")
    steal_dataset = StealDataset.build(
            orig_steal_dataset, args.surrogate_ds_size,
            encoder.model, classifier, device,
            None if inferred_encoder is None else inferred_encoder.model, device_steal)
    logger.info("finished building surrogate dataset")

    stealloader = utils.data.Loader(
        dataset=steal_dataset, batch_size=cfg["batch_size"], train=True)

    embed_trainloader = utils.data.Loader(
        dataset=utils.data.get_pre_embed_dataset(args.trainset_path),
        batch_size=cfg["batch_size"], train=False)
    embed_testloader = utils.data.Loader(
        dataset=utils.data.get_pre_embed_dataset(args.testset_path),
        batch_size=cfg["batch_size"], train=False)

    if pei_assist:
        inferred_trainloader = utils.data.Loader(
            dataset=utils.data.get_pre_embed_dataset(args.inferred_trainset_path),
            batch_size=cfg["batch_size"], train=False)
        inferred_testloader = utils.data.Loader(
            dataset=utils.data.get_pre_embed_dataset(args.inferred_testset_path),
            batch_size=cfg["batch_size"], train=False)
    else:
        inferred_trainloader = utils.data.Loader(
            dataset=utils.data.get_img_dataset(args.dataset, train=True, trans=steal_trans),
            batch_size=cfg["batch_size"], train=False)
        inferred_testloader = utils.data.Loader(
            dataset=utils.data.get_img_dataset(args.dataset, train=False, trans=steal_trans),
            batch_size=cfg["batch_size"], train=False)


    criterion = None
    if args.steal_type == "logit-label":
        criterion = lambda _y, y: torch.nn.functional.mse_loss(_y, y)
    elif args.steal_type == "hard-label":
        criterion = lambda _y, y: torch.nn.functional.cross_entropy(_y, y.argmax(dim=1))
    else:
        raise NotImplementedError

    steps = cfg["steps"]

    for step in range(steps):
        # x, steal_x = next(stealloader)
        # with torch.no_grad():
        #     y = classifier(encoder.model(x.to(device))).to(device_steal)
        #     # embed_x = encoder.model(x.to(device))
        #     # y = classifier(embed_x)
        #     # embed_x = embed_x.to(device_steal)
        #     # y = y.to(device_steal)

        # steal_model.train()
        # if pei_assist:
        #     with torch.no_grad():
        #         embed_x = inferred_encoder.model(steal_x.to(device_steal))
        #     _y = steal_model(embed_x)
        # else:
        #     _y = steal_model(steal_x.to(device_steal))

        x, y = next(stealloader)
        steal_model.train()

        _y = steal_model(x.to(device_steal))
        y = y.to(device_steal)

        optim.zero_grad()
        loss = criterion(_y, y)
        loss.backward()
        optim.step()
        scheduler.step()

        if (step+1) % cfg["eval_freq"] == 0:
            test_acc, test_agree = evaluate(embed_trainloader, embed_testloader,
                                            inferred_trainloader, inferred_testloader,
                                            classifier, steal_model, device, device_steal)

            logger.info("Step [{}/{}]".format(step+1, cfg["steps"]))
            logger.info("Model stealing performance:")
            logger.info("test_acc: {:.2%}, test_agree: {:.2%}".format(test_acc, test_agree))
            # logger.info("test_acc: {:.2%}, test_agree: {:.2%}".format(test_acc, test_agree))

    torch.save(steal_model, os.path.join(args.save_dir, "{}_steal-model.pt".format(args.save_name)))


def main(args, logger):
    cfg = None
    if args.exp_config is not None:
        with open(args.exp_config, "r") as f:
            cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)
        with open(os.path.join(args.save_dir, "{}_cfg.yaml".format(args.save_name)), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    exp_steal(args, cfg, logger)


if __name__ == "__main__":
    args = get_args()
    logger = utils.generic_init(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    try:
        main(args, logger)

    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
