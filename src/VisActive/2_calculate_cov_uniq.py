import os
import pickle as pkl

import src.VisActive._settings as settings
from src.VisActive.TrainingStage.Cov_and_Uniq import coverage_and_uniqueness
from src.VisActive.TrainingStage.collect_vc_info import collect_vc_info


output = collect_vc_info(training_path=settings.training_path,
                         val_path=settings.val_path,
                         image_shape=settings.image_shape,
                         batch_size=settings.batch_size,
                         model_path=settings.hvc_model_path,
                         neural_network=settings.neural_network_hvc,
                         num_classes=settings.num_classes,
                         weight_decay=settings.weight_decay,
                         drop_out=0,
                         num_vc=settings.num_vc,
                         p_h=settings.p_h,
                         p_w=settings.p_w,
                         p_d=settings.p_d)

# for k, v in output.items():
#     print(k, v)

coverage, uniqueness = coverage_and_uniqueness(output,
                                               num_classes=settings.num_classes,
                                               num_vc=settings.num_vc,
                                               vc_per_class=settings.vc_per_class,
                                               top_k=settings.gama)


# Save output, coverage, and uniqueness
with open(os.path.join(settings.output_search_directory, f'vc_info.pkl'), 'wb') as f:
    pkl.dump(output, f, pkl.HIGHEST_PROTOCOL)
with open(os.path.join(settings.output_search_directory, f'coverage.pkl'), 'wb') as f:
    pkl.dump(coverage, f, pkl.HIGHEST_PROTOCOL)
with open(os.path.join(settings.output_search_directory, f'uniqueness.pkl'), 'wb') as f:
    pkl.dump(uniqueness, f, pkl.HIGHEST_PROTOCOL)
