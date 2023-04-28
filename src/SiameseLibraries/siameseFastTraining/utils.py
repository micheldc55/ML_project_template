def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unfreeze_layers(model, num_layers_to_unfreeze):
    ct = 0
    children_list = list(model.children())
    for child in reversed(children_list):
        ct += 1
        if ct <= num_layers_to_unfreeze:
            for param in child.parameters():
                param.requires_grad = True
        else:
            break
