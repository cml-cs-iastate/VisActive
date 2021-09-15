import os
import time


def save_variables_and_metagraph(sess, saver, model_dir, model_name, step, epoch, use_step=False):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()

    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    if use_step:
        saver.save(sess, checkpoint_path, global_step=epoch, write_meta_graph=False)
    else:
        saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    print(checkpoint_path)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    print(metagraph_filename)
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)


