import numpy as np
from tensorflow.keras import models

from src.VisActive.Utils.data_loader import generator_loader
from src.VisActive.Utils.utils import load_hvc_model, get_file_name_from_path, get_ordered_file_names


def collect_vc_info(training_path, val_path, image_shape, batch_size,
                    model_path, neural_network, num_classes, weight_decay, drop_out, num_vc, p_h, p_w, p_d):
    """
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Loading data and initialize variables

    # Create training generator
    _, _, train_generator = generator_loader(training_path,
                                             val_path,
                                             training_path,
                                             image_shape,
                                             image_shape,
                                             batch_size)

    epoch_size = train_generator.samples // train_generator.batch_size

    # Initialize the result dictionary
    # output = {'img_name': {'class': 1, 'prediction': 1, 'sorted_vc': [0, 1, 2, 3, 4]}}

    output = dict()

    results = dict()
    results['img_name'] = []
    results['class'] = []
    results['prediction'] = []
    results['sorted_vc'] = []

    # ------------------------------------------------------------------------------------------------------------------
    # Model

    # Load model
    model = load_hvc_model(model_path=model_path,
                           neural_network=neural_network,
                           num_classes=num_classes,
                           image_size=image_shape,
                           weight_decay=weight_decay,
                           drop_out=0,
                           num_vc=num_vc,
                           p_h=p_h,
                           p_w=p_w,
                           p_d=p_d)
    # print(model.summary(line_length=300))

    # Build hvc_model to get the VC output
    hvc_model = models.Model(inputs=model.input, outputs=model.get_layer('vcl').output)
    # print(hvc_model.summary(line_length=300))

    # ------------------------------------------------------------------------------------------------------------------
    # Generating VC

    # Iterate each batch in one epoch
    for counter in range(epoch_size):
        input_x, input_y = next(train_generator)

        # Get file name from file paths
        image_file_names = [get_file_name_from_path(f) for f in get_ordered_file_names(train_generator)]

        # Save file names
        results['img_name'].extend(image_file_names)

        # Save ground truth labels
        results['class'].extend(input_y)

        # Predict the class labels
        input_y_predictions = model.predict(input_x)

        # Save predictions
        results['prediction'].extend(input_y_predictions)

        # Run the hvc_model to get the output of the VCL
        vcl_output = hvc_model.predict(input_x)

        vcl_activation = vcl_output[0]

        results['sorted_vc'].extend(vcl_activation)

        # --------------------------------------------------------------------------------------------------------------
        # Generate output dict

        for idx in range(len(results['img_name'])):
            file_name = results['img_name'][idx]
            output[file_name] = dict()
            output[file_name]['class'] = np.argmax(results['class'][idx])
            output[file_name]['prediction'] = np.argmax(results['prediction'][idx])
            output[file_name]['sorted_vc'] = np.argsort(results['sorted_vc'][idx])[::-1]

    return output

