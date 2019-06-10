from graph.loss.loss import *


def eval_true_fake(dis, gen, data_loader, device, config, max_batch=None):
    dis.eval()
    gen.eval()

    cnt = 0
    unl_acc, gen_acc, max_unl_acc, max_gen_acc = 0., 0., 0., 0.
    for i, (images, _) in enumerate(data_loader.get_iter()):
        with torch.no_grad():
            images = images.to(device)
            noise = torch.Tensor(images.size(0), config['model']['noise_size']).uniform_().to(device)

        unl_feat = dis(images, feat=True)  # use median feature outputs
        gen_feat = dis(gen(noise), feat=True)

        unl_logits = dis.out_net(unl_feat)
        gen_logits = dis.out_net(gen_feat)

        unl_logsumexp = log_sum_exp(unl_logits)
        gen_logsumexp = log_sum_exp(gen_logits)

        # true-fake accuracy
        # gt 0.5 means inputs is larger than 0
        unl_acc += torch.mean(torch.sigmoid(unl_logsumexp).gt(0.5).float()).item()
        gen_acc += torch.mean(torch.sigmoid(gen_logsumexp).gt(0.5).float()).item()
        # top-1 logit compared to 0: to verify Assumption (2) and (3)
        max_unl_acc += torch.mean(unl_logits.max(1)[0].gt(0.0).float()).item()
        max_gen_acc += torch.mean(gen_logits.max(1)[0].gt(0.0).float()).item()

        cnt += 1
        if max_batch is not None and i >= max_batch - 1:
            break

    return float(unl_acc) / cnt, float(gen_acc) / cnt, float(max_unl_acc) / cnt, float(max_gen_acc) / cnt


def eval_classification(dis, gen, data_loader, device, max_batch=None):
    dis.eval()
    gen.eval()

    loss, incorrect, cnt = 0, 0, 0
    for i, (images, labels) in enumerate(data_loader.get_iter()):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
        pred_prob = dis(images)
        loss += d_criterion(pred_prob, labels).item() * images.shape[0]
        cnt += images.shape[0]
        incorrect += torch.ne(torch.max(pred_prob, 1)[1], labels).data.sum()
        if max_batch is not None and i >= max_batch - 1:
            break
    average_loss = float(loss) / float(cnt)
    error_rate = float(incorrect) / float(cnt)
    return average_loss, error_rate, int(incorrect)
