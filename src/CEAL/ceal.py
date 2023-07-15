import numpy as np
from tensorflow.keras import callbacks, utils

import src.CEAL.settings as settings
from src.VisActive.Utils.data_loader import generator_loader
from src.CEAL.utils import initialize_model, get_high_confidence_samples, get_uncertain_samples, generate_balanced_batch, get_image_name, copy_images, count


def run_ceal():

    # ------------------------------------------------------------------------------------------------------------------
    # Load dataset

    # Create training generator
    train_generator, test_generator, search_generator = generator_loader(settings.training_path,
                                                                         settings.testing_path,
                                                                         settings.search_path,
                                                                         settings.image_shape,
                                                                         settings.image_shape,
                                                                         settings.batch_size)

    x_initial, y_initial = generate_balanced_batch(train_generator, int(train_generator.samples / train_generator.batch_size) + 1)
    x_test, y_test = generate_balanced_batch(test_generator, int(test_generator.samples / test_generator.batch_size) + 1)
    x_pool, y_pool = generate_balanced_batch(search_generator, int(search_generator.samples / search_generator.batch_size) + 1)

    unlabeled_image_file_name = get_image_name(search_generator, int(search_generator.samples / search_generator.batch_size) + 1)
    # print(unlabeled_image_file_name[:10])

    # ------------------------------------------------------------------------------------------------------------------
    # Build and train the model

    early_stop_ = callbacks.EarlyStopping(monitor='val_loss', patience=1)
    checkpoint = callbacks.ModelCheckpoint(settings.base_model_path, monitor='val_accuracy', mode='max', save_best_only=True)

    model = initialize_model(x_initial, y_initial, x_test, y_test, checkpoint)

    # ------------------------------------------------------------------------------------------------------------------
    # CEAL

    w, h, c = x_pool[-1, ].shape

    # unlabeled samples
    unlabeled_samples_ = x_pool, y_pool

    # initially labeled samples
    initially_labeled_samples_ = x_initial, y_initial

    # high confidence samples
    high_confidence_samples_ = np.empty((0, w, h, c)), np.empty((0, settings.num_classes))

    for i in range(1, settings.maximum_iterations):

        y_pred_prob = model.predict(unlabeled_samples_[0], verbose=settings.verbose_)

        # Get the least certain samples (The ones with low conf score)
        _, un_idx = get_uncertain_samples(y_pred_prob,
                                          settings.uncertain_samples_size,
                                          criteria=settings.uncertain_criteria)

        # Copy the unlabeled images with low conf to the labeled image dataset
        initially_labeled_samples_ = np.append(initially_labeled_samples_[0], np.take(unlabeled_samples_[0], un_idx, axis=0), axis=0), np.append(initially_labeled_samples_[1], np.take(unlabeled_samples_[1], un_idx, axis=0), axis=0)

        # Get the most certain samples (The ones with high conf score)
        if settings.cost_effective:
            hc_idx, hc_labels = get_high_confidence_samples(y_pred_prob, settings.delta)
            # remove samples that has been selected through uncertain stage
            hc = np.array([[i, l] for i, l in zip(hc_idx, hc_labels) if i not in un_idx])
            if hc.size != 0:
                high_confidence_samples_ = np.take(unlabeled_samples_[0], hc[:, 0], axis=0), utils.to_categorical(hc[:, 1], settings.num_classes)

        if i % settings.fine_tunning_interval == 0:
            d_train_x = np.concatenate((initially_labeled_samples_[0], high_confidence_samples_[0])) if high_confidence_samples_[0].size != 0 else initially_labeled_samples_[0]
            d_train_y = np.concatenate((initially_labeled_samples_[1], high_confidence_samples_[1])) if high_confidence_samples_[1].size != 0 else initially_labeled_samples_[1]

            model.fit(d_train_x,
                      d_train_y,
                      validation_data=(x_test, y_test),
                      batch_size=settings.batch_size,
                      shuffle=True,
                      epochs=settings.epochs,
                      verbose=settings.verbose_,
                      callbacks=[early_stop_])

            settings.delta -= (settings.threshold_decay * settings.fine_tunning_interval)

        # remove the moved uncertain images from the unlabeled dataset
        unlabeled_samples_ = np.delete(unlabeled_samples_[0], un_idx, axis=0), np.delete(unlabeled_samples_[1], un_idx, axis=0)
        high_confidence_samples_ = np.empty((0, w, h, c)), np.empty((0, settings.num_classes))

        # --------------------------------------------------------------------------------------------------------------
        # Evaluate
        _, acc = model.evaluate(x_test,
                                y_test,
                                batch_size=settings.batch_size,
                                verbose=settings.verbose_)

        print(f'Iteration: {i}; High Confidence Samples: {len(high_confidence_samples_[0])};'
              f' Uncertain Samples: {len(initially_labeled_samples_[0])};'
              f' Delta: {settings.delta};'
              f' Labeled Dataset Size: {len(initially_labeled_samples_[0])};'
              f' Accuracy: {acc}')

        # --------------------------------------------------------------------------------------------------------------
        # Copy the uncertain labeled image to the labeled dataset

        images_to_be_copied = list()
        for idx in un_idx:
            images_to_be_copied.append(unlabeled_image_file_name[idx])

        # Remove the image file names from the list
        for idx in un_idx:
            unlabeled_image_file_name[idx] = '*****'
        for idx in range(len(unlabeled_image_file_name) - 1, -1, -1):
            if unlabeled_image_file_name[idx] == '*****':
                unlabeled_image_file_name.pop(idx)

        # copy_images(images_to_be_copied, iteration=i)
        count_rare, count_common = count(images_to_be_copied)
        print(f'Iteration: {i}, count_rare: {count_rare}')


if __name__ == '__main__':
    run_ceal()
