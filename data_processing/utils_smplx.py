
import torch
import smplx


def get_smplx_models(model_folder):
    smplx_model_male = smplx.create(model_folder, model_type='smplx',
                                gender='male',
                                ext='npz',
                                num_betas=10,
                                flat_hand_mean=True,
                                use_pca=False)

    smplx_model_female = smplx.create(model_folder, model_type='smplx',
                                    gender='female',
                                    ext='npz',
                                    num_betas=10,
                                    flat_hand_mean=True,
                                    use_pca=False)

    smplx_model_neutral = smplx.create(model_folder, model_type='smplx',
                                    gender='neutral',
                                    ext='npz',
                                    flat_hand_mean=True,
                                    num_betas=11,
                                    use_pca=False)

    return {"male": smplx_model_male, "female": smplx_model_female, "neutral": smplx_model_neutral}


def get_smplx_vertices(poses, betas, trans, gender, smplx_models):
    assert str(gender) in ["male", "female", "neutral"], f"gener is {smplx_models[str(gender)]}, which is not male, female or neutral"

    trans_np = trans.detach().numpy() if type(trans) == torch.Tensor else trans
    # model_out = smplx_models[str(gender)](betas=torch.tensor(betas).unsqueeze(0).float(),
    #                         global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
    #                         body_pose=torch.tensor(poses[3:66]).unsqueeze(0).float(),
    #                         left_hand_pose=torch.tensor(poses[75:120]).unsqueeze(0).float(),
    #                         right_hand_pose=torch.tensor(poses[120:165]).unsqueeze(0).float(),
    #                         jaw_pose=torch.tensor(poses[66:69]).unsqueeze(0).float(),
    #                         leye_pose=torch.tensor(poses[69:72]).unsqueeze(0).float(),
    #                         reye_pose=torch.tensor(poses[72:75]).unsqueeze(0).float(),
    #                         transl=torch.tensor(trans).unsqueeze(0))

    model_out = smplx_models[str(gender)](betas=torch.tensor(betas).unsqueeze(0).float(),
                            global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
                            body_pose=torch.tensor(poses[3:66]).unsqueeze(0).float(),
                            left_hand_pose=torch.tensor(poses[75:120]).unsqueeze(0).float(),
                            right_hand_pose=torch.tensor(poses[120:165]).unsqueeze(0).float(),
                            jaw_pose=torch.tensor(poses[66:69]).unsqueeze(0).float(),
                            leye_pose=torch.tensor(poses[69:72]).unsqueeze(0).float(),
                            reye_pose=torch.tensor(poses[72:75]).unsqueeze(0).float(),
                            transl=torch.tensor(trans_np).unsqueeze(0))

    return model_out.vertices[0], model_out.joints[0]