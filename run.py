from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

def get_pose_image_from_file(
    image_path='./images/p1.jpg',
    model='cmu',
    resize='0x0',
    resize_out_ratio=4.0
):
    # argparse 관련 코드를 제거하고, 함수 인자를 사용

    w, h = model_wh(resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    image = common.read_imgfile(image_path, None, None)
    if image is None:
        raise ValueError(f"Image cannot be read from path={image_path}")

    humans = e.inference(image, resize_to_default=(
        w > 0 and h > 0), upsample_size=resize_out_ratio)
    image_with_humans = TfPoseEstimator.draw_humans(
        image, humans, imgcopy=False)

    return image_with_humans