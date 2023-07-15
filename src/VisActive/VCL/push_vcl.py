import numpy as np


# Tested
def push(vc_weights, input_tensor, vc_distance, global_min_vc_distance, labels):
    """
    Push the each prototype to the closest patch of the image that belong to the same label of the VC
    maintain the minimum distance of each prototype with the closest patch
    :param vc_weights: numpy VCL kernel tensor                                                  # shape [p_h, p_w, depth, num_vc]
    :param input_tensor: numpy input of the VCL                                                 # shape [batch_size, height, width, depth]
    :param vc_distance: numpy minimum distance of each prototype with a patch                   # shape [batch_size, height, width, num_vc]
    :param global_min_vc_distance: numpy minimum distance of a patch of each prototype          # shape [num_vc]
    :param labels: numpy labels of the y_true                                                   # shape [batch_size, num_classes]
    """
    vc_shape = vc_weights.shape
    proto_h, proto_w, num_vc = vc_shape[0], vc_shape[1], vc_shape[3]
    batch_size = labels.shape[0]
    num_classes = labels.shape[1]
    vc_per_class = int(num_vc // num_classes)

    old_vc_weights = np.copy(vc_weights)
    old_global_min_vc_distance = np.copy(global_min_vc_distance)

    # Group the image indices based on their class label
    class_to_img_index_dict = {key: [] for key in range(num_classes)}  # {0:[], 1:[], ..., 4:[]}
    for img_index in range(batch_size):
        img_label = np.argmax(labels[img_index], axis=0)
        class_to_img_index_dict[img_label].append(img_index)

    seen_images = {i: [] for i in range(batch_size)}

    # Iterate for every prototype and check if there is a patch in the current batch closer than the one in the previous
    # batches. If so, update the global_min_fmap_patches and then return in.
    for prototype_idx in range(num_vc):

        # target_class is the class of the prototype prototype_idx belongs to
        target_class = int(prototype_idx // vc_per_class)

        # if there is not images of the target_class from this batch we go on to the next prototype
        if len(class_to_img_index_dict[target_class]) == 0:
            continue

        # This line first get the images that are related to that VC, and then get the VC that related
        # to the class of those images
        # shape [batch_size, height, width, num_vc] --> # shape [num_of_images_of_target_class, height, width]
        prototype_idx_distance = vc_distance[class_to_img_index_dict[target_class]][:, :, :, prototype_idx]

        batch_min_vc_idx_distance = np.amin(prototype_idx_distance)

        if batch_min_vc_idx_distance < global_min_vc_distance[prototype_idx]:

            # batch_argmin_vc_idx_distance will hold the full index to the closest patch [1, 1, 1, :] so that we
            # retrieve the actual patch closest to prototype prototype_idx
            batch_argmin_vc_idx_distance = list(np.unravel_index(np.argmin(prototype_idx_distance, axis=None), prototype_idx_distance.shape))

            # update the actual image index in the batch (the one already in there is the index in class_to_img_index_dict dict)
            batch_argmin_vc_idx_distance[0] = class_to_img_index_dict[target_class][batch_argmin_vc_idx_distance[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_vc_idx_distance[0]
            fmap_height_start_index = batch_argmin_vc_idx_distance[1]
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_vc_idx_distance[2]
            fmap_width_end_index = fmap_width_start_index + proto_w

            # shape [1, 1, depth]
            batch_min_feature_map_patch_idx = input_tensor[img_index_in_batch, fmap_height_start_index:fmap_height_end_index, fmap_width_start_index:fmap_width_end_index, :]

            if len(seen_images[img_index_in_batch]) == 0:
                vc_weights[:, :, :, prototype_idx] = batch_min_feature_map_patch_idx
                global_min_vc_distance[prototype_idx] = batch_min_vc_idx_distance
                seen_images[img_index_in_batch].append((fmap_height_start_index, fmap_width_start_index, batch_min_vc_idx_distance, prototype_idx))

            else:
                flag = True
                count_ = 0
                for s in seen_images[img_index_in_batch]:
                    if s[0] == fmap_height_start_index and s[1] == fmap_width_start_index:
                        flag = False
                        break
                    count_ += 1

                if flag:
                    vc_weights[:, :, :, prototype_idx] = batch_min_feature_map_patch_idx
                    global_min_vc_distance[prototype_idx] = batch_min_vc_idx_distance
                    seen_images[img_index_in_batch].append((fmap_height_start_index, fmap_width_start_index, batch_min_vc_idx_distance, prototype_idx))

                else:
                    if seen_images[img_index_in_batch][count_][2] > batch_min_vc_idx_distance:
                        vc_weights[:, :, :, prototype_idx] = batch_min_feature_map_patch_idx
                        global_min_vc_distance[prototype_idx] = batch_min_vc_idx_distance
                        seen_images[img_index_in_batch].append((fmap_height_start_index, fmap_width_start_index, batch_min_vc_idx_distance, prototype_idx))

                        p_index = seen_images[img_index_in_batch][count_][3]
                        vc_weights[:, :, :, p_index] = old_vc_weights[:, :, :, p_index]
                        global_min_vc_distance[p_index] = old_global_min_vc_distance[p_index]

                        del seen_images[img_index_in_batch][count_]

    return [vc_weights, global_min_vc_distance]
