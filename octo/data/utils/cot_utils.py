import enum


class CotTag(enum.Enum):
    GRIPPER_POSITION = "GRIPPER POSITION:"


def abbreviate_tag(tag: str):
    return tag[0] + tag[-2]


def get_cot_tags_list():
    return [
        CotTag.GRIPPER_POSITION.value,
    ]


def get_cot_database_keys():
    return {
        CotTag.GRIPPER_POSITION.value: "gripper",
    }