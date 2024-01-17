from jax.experimental import jax2tf
from jax import numpy as jnp
from functools import partial

import numpy as np
import tensorflow as tf
from typing import Dict, Any
import jax
import flax
from octo.model.octo_model import OctoModel


#################### HELPERS FOR SAVING ###############################


def model_jax(model, params, inputs, default_kwargs):
    return model.replace(params=params).sample_actions(
        **inputs,
        **default_kwargs,
    )


ExampleInputs = Any
def create_tf_model(model: OctoModel, possible_input_patterns: Dict[str, ExampleInputs], default_kwargs):
    params_vars = tf.nest.map_structure(tf.Variable, model.params)
    pred_fn = partial(model_jax, model, default_kwargs=default_kwargs)
    prediction_tf = lambda inputs: jax2tf.convert(pred_fn)(params_vars, inputs)
    tf_model = tf.Module()
    # Tell the model saver what are the variables.
    tf_model._variables = tf.nest.flatten(params_vars)
    possible_specs = {
        k: jax.tree_map(lambda x: tf.TensorSpec(shape=x.shape, dtype=tf.as_dtype(x.dtype)), v)
        for k, v in possible_input_patterns.items()
    }

    for k, spec in possible_specs.items():
        print('Adding function named', k, 'with spec', flax.core.pretty_repr(spec), 'to model')
        setattr(tf_model, k, tf.function(prediction_tf, jit_compile=True, autograph=False, input_signature=[spec]))
    return tf_model


###################### MODEL SAVING #############################

from octo.model.octo_model import OctoModel
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")

ws1_obs = {k: model.example_batch['observation'][k][:1, :1] for k in ['image_primary', 'pad_mask']}
ws2_obs = {k: model.example_batch['observation'][k][:1, :2] for k in ['image_primary', 'pad_mask']}

gc_task = {k: model.example_batch['task'][k][:1] for k in ['image_primary']}
lc_task = {k: jax.tree_map(lambda x: x[:1], model.example_batch['task'][k]) for k in ['language_instruction']}

example_rng = np.array(jax.random.PRNGKey(0))

possible_input_patterns = {
    f'{task_type}_ws{ws}': {
        'observations': obs,
        'tasks': task,
        'rng': example_rng,
    }
    for (task_type, task) in [('gc', gc_task), ('lc', lc_task)]
    for (ws, obs) in [(1, ws1_obs), (2, ws2_obs)]
}
default_kwargs = dict(
    train=False,
    argmax=False,
    sample_shape=tuple(),
    temperature=1.0,
)

tf_model = create_tf_model(model, possible_input_patterns, default_kwargs=default_kwargs)
tf.saved_model.save(tf_model, 'octo_base')


###################### EXAMPLE MODEL LOADING #############################

restored_model = tf.saved_model.load('octo_base')
restored_model.gc_ws1({
    'observations': {
        'image_primary': np.zeros((1, 1, 256, 256, 3), dtype=np.float32),
        'pad_mask': np.ones((1, 1,), dtype=np.bool_),
    },
    'tasks': {
        'image_primary': np.zeros((1, 256, 256, 3), dtype=np.float32),
    },
    'rng': np.zeros((2,), dtype=np.uint32),
})
