from tensorflow.keras import models

from src.VisActive.Utils.data_loader import generator_loader
from src.VisActive.Utils.utils import load_hvc_model, get_file_name_from_path, get_ordered_file_names
from src.VisActive.Utils.plot_rect import visualize_vc_rect


def vc_interpretation(training_path, search_path, image_shape, batch_size,
                      model_path, neural_network, num_classes, weight_decay, drop_out, num_vc, p_h, p_w, p_d):
    """
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Loading data and initialize variables

    # Create Training, validation, and Testing generators
    _, search_generator, _ = generator_loader(training_path,
                                              search_path,
                                              None,
                                              image_shape,
                                              image_shape,
                                              batch_size)

    # Calculate the epoch size in terms of number of batches
    epoch_size = search_generator.samples // search_generator.batch_size

    # ------------------------------------------------------------------------------------------------------------------
    # Model

    # Load model
    model = load_hvc_model(model_path=model_path,
                           neural_network=neural_network,
                           num_classes=num_classes,
                           image_size=image_shape,
                           weight_decay=weight_decay,
                           drop_out=drop_out,
                           num_vc=num_vc,
                           p_h=p_h,
                           p_w=p_w,
                           p_d=p_d)
    print(model.summary())

    # Build hvc_model to get the VC output
    hvc_model = models.Model(inputs=model.input, outputs=model.get_layer('vcl').output)
    print(hvc_model.summary())

    # ------------------------------------------------------------------------------------------------------------------
    # Generating VC

    # Iterate each batch in one epoch
    for counter in range(epoch_size):
        input_x, _ = next(search_generator)

        # Get file name from file paths
        image_file_names = [get_file_name_from_path(f) for f in get_ordered_file_names(search_generator)]

        # Run the hvc_model to get the output of the VCL
        vcl_output = hvc_model.predict(input_x)

        num_vc = vcl_output[1].shape[-1]
        coordinates = vcl_output[4]

        # Generate heatmap and save image patches
        visualize_vc_rect(input_x, coordinates, num_vc, image_file_names)


import src.VisActive._settings as settings

vc_interpretation(training_path=settings.training_path,
                  search_path=settings.search_path,
                  image_shape=settings.image_shape,
                  batch_size=settings.batch_size,
                  model_path=settings.hvc_model_path,
                  neural_network=settings.neural_network,
                  num_classes=settings.num_classes,
                  weight_decay=settings.weight_decay,
                  drop_out=0,
                  num_vc=settings.num_vc,
                  p_h=settings.p_h,
                  p_w=settings.p_w,
                  p_d=settings.p_d)
