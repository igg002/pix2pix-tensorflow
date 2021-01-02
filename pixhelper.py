import os
import shutil

# PixHelper: Helper class for the pix2pix training ipython notebook
class PixHelper:
  # ============= FILE MANAGEMENT =============
  # Clear out the passed in list of directories
  @staticmethod
  def clear_dirs(dirs):
    for dir in dirs:
      if os.path.exists(dir):
        shutil.rmtree(dir)


  # Make the passed directory after checking if it exists
  @staticmethod
  def check_make_dir(dir):
    if not os.path.exists(dir):
      os.makedirs(dir)
      print('Path created: {}'.format(dir))


  # ============= COMMAND BUILDING =============
  # Build command with the passed arguments
  @staticmethod
  def parse_command(code_dir, args):
    command = 'python {}/pix2pix.py'.format(code_dir)
    for key, val in args.items():
      command += ' --{} {}'.format(key, val)
    return command
  

  # ============= PIX2PIX =============
  # Download sample dataset of the passed title
  @staticmethod
  def download_sample_dataset(code_dir, title):
    print('Starting download...')
    command = 'python {}/tools/download-dataset.py {}'.format(code_dir, title)
    exit_code = os.system(command)
    print('Download complete with exit code {}'.format(exit_code))


  # Preprocess data points
  @staticmethod
  def preprocess(code_dir, a_dir, b_dir, output_dir, exec=None):
    # Check if the number of datapoints match
    a_items, b_items = os.listdir(a_dir), os.listdir(b_dir)
    a_len = len([name for name in a_items])
    b_len = len([name for name in b_items])
    assert(a_len == b_len)

    # Change names
    for i, (a, b) in enumerate(zip(sorted(a_items), sorted(b_items))):
      a_path, b_path = os.path.join(a_dir, a), os.path.join(b_dir, b)
      ext = os.path.splitext(a)[1]
      os.rename(a_path, os.path.join(a_dir, str(i) + ext))
      os.rename(b_path, os.path.join(b_dir, str(i) + ext))

    # Combine
    combination_command = (
      'python {}/tools/process.py '
      '--input_dir {} '
      '--b_dir {} '
      '--operation combine '
      '--output_dir {}'
    )
    combination_command = combination_command.format(code_dir, a_dir, b_dir, output_dir)

    # Split
    split_command = (
      'python {}/tools/split.py '
      '--dir {}'
    )
    split_command = split_command.format(code_dir, output_dir)

    # Execute if executor is passed
    commands = (combination_command, split_command)
    if exec is not None:
      exec(commands)
      return None

    # Return command tuple
    return commands


  # Export checkpoint
  @staticmethod
  def export_model(code_dir, checkpoint_dir, output_dir, exec=None):
    path_head = os.path.split(output_dir)[0]
    if not os.path.exists(path_head):
      os.mkdir(path_head)

    command = 'python {}/server/tools/export-checkpoint.py --checkpoint {} --output_file {}'
    command = command.format(code_dir, checkpoint_dir, output_dir)
    
    # Execute if executor is passed
    if exec is not None:
      exec(command)
      return None

    # Return command tuple
    return command