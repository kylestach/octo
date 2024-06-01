"""Defines dataset mixtures and weights for the Open X-Embodiment Datasets."""

BRIDGE_MIX = [
    ("bridge_dataset", 1.0),
]

RT_X_MIX = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("taco_extra", 2.0),
    ("jaco_play", 2.0),
    ("berkeley_cable_routing", 3.0),
    ("roboturk", 1.0),
    ("nyu_door_opening_surprising_effectiveness", 5.0),
    ("viola", 2.0),
    ("berkeley_autolab_ur5", 1.0),
    ("toto", 1.0),
]


OXE_FRANKA_MIX = [
    ("taco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("viola", 1.0),
    ("toto", 1.0),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 3.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
    ("maniskill_dataset_converted_externally_to_rlds", 0.1),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 5.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("kaist_nonprehensile_converted_externally_to_rlds", 3.0),
    ("stanford_robocook_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    ("utaustin_mutex", 1.0),
    # ("cmu_playing_with_food", 1.0),
    ("cmu_play_fusion", 1.0),
]


OXE_MAGIC_SOUP = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("taco_extra", 2.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 2.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 2.0),
    ("berkeley_autolab_ur5", 2.0),
    ("toto", 1.0),
    ("language_table", 0.1),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    # ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0), --> weird actions
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 0.2),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    # ("uiuc_d3field", 1.0),  --> somehow raw data is broken
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 2.0),
    ("cmu_stretch", 1.0),
]

OXE_MAGIC_SOUP_BALANCED = [
    ("kuka", 0.14503701874493363),
    ("taco_play", 0.06657998827701668),
    ("taco_extra", 0.015452958868388737),
    ("jaco_play", 0.010914534155076169),
    ("berkeley_cable_routing", 0.005925612796973822),
    ("roboturk", 0.052499238268860826),
    ("nyu_door_opening_surprising_effectiveness", 0.0028565519070650833),
    ("viola", 0.021369612129854),
    ("berkeley_autolab_ur5", 0.027421498380401588),
    ("toto", 0.045595496181288435),
    ("language_table", 0.09863155061985435),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 0.10030032010542056),
    ("austin_buds_dataset_converted_externally_to_rlds", 0.004775432426062442),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 0.01884652293499813),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.05526993262706029),
    ("austin_sailor_dataset_converted_externally_to_rlds", 0.04943059735717906),
    ("austin_sirius_dataset_converted_externally_to_rlds", 0.03918942829266809),
    ("bc_z", 0.14503701874493363),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 0.00124985520344411),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 0.020472678629801757),
    ("utaustin_mutex", 0.05066099356944051),
    ("berkeley_fanuc_manipulation", 0.017530731149920712),
    ("cmu_stretch", 0.003502058441908362),
    ("droid", 0.001450370187449336),
]


OXE_FLEX_ACT_SOUP = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("taco_extra", 2.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 2.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 2.0),
    ("berkeley_autolab_ur5", 2.0),
    ("toto", 1.0),
    ("language_table", 0.1),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 0.2),
    ("berkeley_mvp_converted_externally_to_rlds", 1.0),
    # ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    # ("uiuc_d3field", 1.0),  --> somehow raw data is broken
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 2.0),
    ("cmu_stretch", 1.0),
    ("omnimimic_gnm_dataset", 1.0),
    ("aloha_static_dataset", 3.0),
    # ("aloha_dagger_dataset", 1.0),
    ("aloha_mobile_dataset", 2.0),
    # ("fmb_dataset", 1.0),
    ("dobbe", 1.0),
    ("roboset", 0.5),
    ("rh20t", 0.5),
]


OXE_EXPANDED_SOUP = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.4),
    ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("taco_extra", 2.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 2.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 1.0),
    ("berkeley_autolab_ur5", 2.0),
    ("toto", 1.0),
    ("language_table", 0.1),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    # ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0), --> weird actions
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 0.2),
    # ("berkeley_mvp_converted_externally_to_rlds", 1.0), # JOINT_POS
    # ("berkeley_rpt_converted_externally_to_rlds", 1.0), # JOINT_POS
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    # ("uiuc_d3field", 1.0),  --> somehow raw data is broken
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 2.0),
    ("cmu_stretch", 1.0),
    # ("gnm_dataset", 0.5), # NAV_2D
    # ("aloha_static_dataset", 3.0), # JOINT_POS_BIMANUAL
    # ("aloha_dagger_dataset", 1.0), # JOINT_POS_BIMANUAL
    # ("aloha_mobile_dataset", 1.0), # JOINT_POS_BIMANUAL
    # ("fmb_dataset", 1.0),  --> doesn't exist?
    ("dobbe", 0.3),
    # ("roboset", 0.1), # JOINT_POS
    ("rh20t", 0.1),
]

OXE_FULL_MIX = [
    ("fractal20220817_data", 1.0),
    ("kuka", 1.0),
    ("bridge_dataset", 1),
    ("taco_play", 1.0),
    ("taco_extra", 1.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 1.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 1.0),
    ("berkeley_autolab_ur5", 1.0),
    ("toto", 1.0),
    ("language_table", 1.0),
    ("columbia_cairlab_pusht_real", 1.0),
    ("stanford_kuka_multimodal_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_rot_dataset_converted_externally_to_rlds", 1.0),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 1.0),
    ("maniskill_dataset_converted_externally_to_rlds", 1.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
    ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 1.0),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 1.0),
    ("ucsd_pick_and_place_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 1.0),
    ("utokyo_pr2_opening_fridge_converted_externally_to_rlds", 1.0),
    ("utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds", 1.0),
    ("utokyo_xarm_pick_and_place_converted_externally_to_rlds", 1.0),
    ("utokyo_xarm_bimanual_converted_externally_to_rlds", 1.0),
    ("robo_net", 1.0),
    ("berkeley_mvp_converted_externally_to_rlds", 1.0),
    ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("kaist_nonprehensile_converted_externally_to_rlds", 1.0),
    ("stanford_mask_vit_converted_externally_to_rlds", 1.0),
    ("tokyo_u_lsmo_converted_externally_to_rlds", 1.0),
    ("dlr_sara_pour_converted_externally_to_rlds", 1.0),
    ("dlr_sara_grid_clamp_converted_externally_to_rlds", 1.0),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("asu_table_top_converted_externally_to_rlds", 1.0),
    ("stanford_robocook_converted_externally_to_rlds", 1.0),
    ("imperialcollege_sawyer_wrist_cam", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    ("uiuc_d3field", 1.0),
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 1.0),
    ("cmu_playing_with_food", 1.0),
    ("cmu_play_fusion", 1.0),
    ("cmu_stretch", 1.0),
    ("omnimimic_gnm_dataset", 1.0),
]

CROSS_EMBODIMENT_TARGET = [
    ("aloha_pen_uncap_diverse_dataset", 0.1),
    ("aloha_dough_cut_dataset", 1 / 60),
    ("aloha_lucy_dataset", 1 / 60),
    ("aloha_drawer_dataset", 1 / 60),
    ("aloha_pick_place_dataset", 1 / 60),
    ("aloha_static_dataset", 1 / 60),
    ("aloha_sushi_cut_full_dataset", 1 / 60),
    ("bridge_dataset", 0.2),
    ("go1", 0.1),
    ("droid_wipe", 0.1),
    ("omnimimic_gnm_dataset", 0.2),
    ("fractal20220817_data", 0.2),
]

ALOHA_MIX = [
    ("aloha_pen_uncap_diverse_dataset", 1 / 2),
    ("aloha_dough_cut_dataset", 1 / 12),
    ("aloha_lucy_dataset", 1 / 12),
    ("aloha_drawer_dataset", 1 / 12),
    ("aloha_pick_place_dataset", 1 / 12),
    ("aloha_static_dataset", 1 / 12),
    ("aloha_sushi_cut_full_dataset", 1 / 12),
]

CROSS_EMBODIMENT = [
    (name, weight * 0.3) for name, weight in OXE_MAGIC_SOUP_BALANCED
] + [(name, weight * 0.7) for name, weight in CROSS_EMBODIMENT_TARGET]


OXE_NAMED_MIXES = {
    "bridge": BRIDGE_MIX,
    "rtx": RT_X_MIX,
    "rtx_franka": RT_X_MIX + OXE_FRANKA_MIX,
    "oxe_magic_soup": OXE_MAGIC_SOUP,
    "oxe_expanded_soup": OXE_EXPANDED_SOUP,
    "oxe_flex_act_soup": OXE_FLEX_ACT_SOUP,
    "cross_embodiment_target": CROSS_EMBODIMENT_TARGET,
    "aloha_mix": ALOHA_MIX,
    "cross_embodiment": CROSS_EMBODIMENT,
}
