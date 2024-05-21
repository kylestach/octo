import flax
import logging
import numpy as np
import jax.numpy as jnp
import tensorflow as tf


def resnet_26_loader(
    params, restore_path="gs://vit_models/augreg/R26_S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz"
):
    # load pre-trained resnet from: github.com/google-research/vision_transformer/
    with tf.io.gfile.GFile(restore_path, "rb") as f:
        resnet_params = np.load(f)

    resnet_params = {
        tuple(k.split('/')): jnp.array(v)
        for k, v in resnet_params.items()
        if k.startswith('block') or k.startswith('conv_root') or k.startswith('gn_root')
    }

    marked_keys = set()
    flat_params = flax.traverse_util.flatten_dict(params)
    for k in flat_params:
        if len(k) < 3 or k[2] != 'ResNet26FILM_0' or 'Film' in k[3]:
            continue

        new_key = resnet_params[k[3:]]
        if 'gn' in k[-2]:
            new_key = new_key.squeeze()
        elif k[3] == 'conv_root':
            assert flat_params[k].shape[2] % new_key.shape[2] == 0
            conv_tile = int(flat_params[k].shape[2] // new_key.shape[2])
            if conv_tile:
                new_key = jnp.tile(new_key, (1, 1, conv_tile, 1))

        assert new_key.shape == flat_params[k].shape
        assert new_key.dtype == flat_params[k].dtype
        flat_params[k] = new_key
        marked_keys.add(k[3:])

    logging.info("Restored ResNet26 encoder blocks")

    missing_keys = set(resnet_params.keys()) - marked_keys
    assert missing_keys == set(), f"Missing keys: {missing_keys}"

    updated_params = flax.traverse_util.unflatten_dict(flat_params)
    return updated_params
