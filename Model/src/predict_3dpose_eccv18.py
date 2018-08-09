
"""Predicting 3d poses from 2d joints"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py
import copy

import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import procrustes

import viz
import cameras
import data_utils
import linear_model
import csv


tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
tf.app.flags.DEFINE_float("dropout", 1.0, "Dropout keep probability. 1 means no dropout")
tf.app.flags.DEFINE_integer("batch_size", 7025, "Batch size to use during training")
tf.app.flags.DEFINE_integer("epochs", 600, "How many epochs we should train for")
tf.app.flags.DEFINE_boolean("camera_frame", False, "Convert 3d poses to camera coordinates")
tf.app.flags.DEFINE_boolean("max_norm", True  , "Apply maxnorm constraint to the weights")
tf.app.flags.DEFINE_boolean("batch_norm", True, "Use batch_normalization")
tf.app.flags.DEFINE_boolean("centering_2d", False, "Use centering 2d around root")

# Data loading
tf.app.flags.DEFINE_boolean("predict_14", False, "predict 14 joints")
tf.app.flags.DEFINE_boolean("use_sh", False, "Use 2d pose predictions from StackedHourglass")
tf.app.flags.DEFINE_string("action","All", "The action to train on. 'All' means all the actions")

# Architecture
tf.app.flags.DEFINE_integer("linear_size", 2048, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("residual", True, "Whether to add a residual connection every 2 layers")

# Evaluation
tf.app.flags.DEFINE_boolean("procrustes", False, "Apply procrustes analysis at test time")
tf.app.flags.DEFINE_boolean("evaluateActionWise",False, "The dataset to use either h36m or heva")

# Directories
tf.app.flags.DEFINE_string("cameras_path","data/h36m/cameras.h5","Directory to load camera parameters")
tf.app.flags.DEFINE_string("data_dir",   "data/h36m_eccv18_challenge/", "Data directory")  #data/h36m_muzi data/h36m_eccv18_challenge/
tf.app.flags.DEFINE_string("detector_2d",   "cpm", "2D pose detector name") #GT_pose_2d   cpm
tf.app.flags.DEFINE_string("train_dir", "experiments_eccv18", "Training directory.")
tf.app.flags.DEFINE_string("prediction_dir", "eccv18_out/", "3D prediction directory")


# Train or load
tf.app.flags.DEFINE_string("mode", 'train', "Experiment mode") # train / eval / generate
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 2400, "Try to load a previous checkpoint.") #7800 2400



# Misc
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

train_dir = os.path.join( FLAGS.train_dir,
  FLAGS.action,
  'dropout_{0}'.format(FLAGS.dropout),
  'epochs_{0}'.format(FLAGS.epochs) if FLAGS.epochs > 0 else '',
  'lr_{0}'.format(FLAGS.learning_rate),
  'residual' if FLAGS.residual else 'not_residual',
  'depth_{0}'.format(FLAGS.num_layers),
  'linear_size{0}'.format(FLAGS.linear_size),
  'batch_size_{0}'.format(FLAGS.batch_size),
  'procrustes' if FLAGS.procrustes else 'no_procrustes',
  'maxnorm' if FLAGS.max_norm else 'no_maxnorm',
  'batch_normalization' if FLAGS.batch_norm else 'no_batch_normalization',
  '{0}'.format(FLAGS.detector_2d),
  'predict_14' if FLAGS.predict_14 else 'predict_17',
  'center_2d' if FLAGS.centering_2d else 'not_center_2d')

print( train_dir )
summaries_dir = os.path.join( train_dir, "log" ) # Directory for TB summaries

# To avoid race conditions: https://github.com/tensorflow/tensorflow/issues/7448
os.system('mkdir -p {}'.format(summaries_dir))

def create_model( session, actions, batch_size, centering_2d = False, for_eccv18 = False):
  """
  Create model and initialize it or load its parameters in a session

  Args
    session: tensorflow session
    actions: list of string. Actions to train/test on
    batch_size: integer. Number of examples in each batch
  Returns
    model: The created (or loaded) model
  Raises
    ValueError if asked to load a model, but the checkpoint specified by
    FLAGS.load cannot be found.
  """

  model = linear_model.LinearModel(
      FLAGS.linear_size,
      FLAGS.num_layers,
      FLAGS.residual,
      FLAGS.batch_norm,
      FLAGS.max_norm,
      batch_size,
      FLAGS.learning_rate,
      summaries_dir,
      FLAGS.predict_14,
      dtype=tf.float16 if FLAGS.use_fp16 else tf.float32,
      centering_2d = centering_2d,
      for_eccv18 = for_eccv18
   )

  if FLAGS.load <= 0:
    # Create a new model from scratch
    print("Creating model with fresh parameters.")
    session.run( tf.global_variables_initializer() )
    return model


  # Load a previously saved model
  ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")
  print( "train_dir", train_dir )

  if ckpt and ckpt.model_checkpoint_path:
    # Check if the specific checkpoint exists
    if FLAGS.load > 0:
      if os.path.isfile(os.path.join(train_dir,"checkpoint-{0}.index".format(FLAGS.load))):
        ckpt_name = os.path.join( os.path.join(train_dir,"checkpoint-{0}".format(FLAGS.load)) )
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    model.saver.restore( session, ckpt.model_checkpoint_path )
    return model
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

  return model

def train_eccv18():
  """Train a linear model for 3d pose estimation"""

  actions = data_utils.define_actions( FLAGS.action )


  # Load 3d data
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d  = data_utils.read_data_eccv18(
    FLAGS.data_dir, FLAGS.centering_2d, FLAGS.detector_2d, dim=3)


  train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_data_eccv18(
    FLAGS.data_dir, FLAGS.centering_2d, FLAGS.detector_2d, dim=2)

  # Avoid using the GPU if requested
  device_count = {"GPU": 2} if FLAGS.use_cpu else {"GPU": 0}
  with tf.Session(config=tf.ConfigProto(
    device_count=device_count,
    allow_soft_placement=True )) as sess:

    print("\n**********************************device_count**********************************\ndevice_count\n\n\n")
    # === Create the model ===
    print("Creating %d bi-layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    model = create_model( sess, actions, FLAGS.batch_size, FLAGS.centering_2d, for_eccv18=True)
    model.train_writer.add_graph( sess.graph )
    print("Model created")

    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
    previous_losses = []

    step_time, loss = 0, 0
    current_epoch = 0
    log_every_n_batches = 100

    for _ in xrange( FLAGS.epochs ):
      current_epoch = current_epoch + 1

      # === Load training batches for one epoch ===
      encoder_inputs, decoder_outputs = model.get_all_batches_eccv18( train_set_2d, train_set_3d, training=True )
      nbatches = len( encoder_inputs )
      print("There are {0} train batches".format( nbatches ))
      start_time, loss = time.time(), 0.

      # === Loop through all the training batches ===
      for i in range( nbatches ):

        if (i+1) % log_every_n_batches == 0:
          # Print progress every log_every_n_batches batches
          print("Working on epoch {0}, batch {1} / {2}... ".format( current_epoch, i+1, nbatches), end="" )

        enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
        step_loss, loss_summary, lr_summary, _ =  model.step( sess, enc_in, dec_out, FLAGS.dropout, isTraining=True )

        if (i+1) % log_every_n_batches == 0:
          # Log and print progress every log_every_n_batches batches
          model.train_writer.add_summary( loss_summary, current_step )
          model.train_writer.add_summary( lr_summary, current_step )
          step_time = (time.time() - start_time)
          start_time = time.time()
          print("done in {0:.2f} ms".format( 1000*step_time / log_every_n_batches ) )

        loss += step_loss
        current_step += 1
        # === end looping through training batches ===

      loss = loss / nbatches
      print("=============================\n"
            "Global step:         %d\n"
            "Learning rate:       %.2e\n"
            "Train loss avg:      %.4f\n"
            "=============================" % (model.global_step.eval(),
            model.learning_rate.eval(), loss) )
      # === End training for an epoch ===

      # === Testing after this epoch ===
      isTraining = False

      n_joints = 17 if not(FLAGS.predict_14) else 14
      encoder_inputs, decoder_outputs = model.get_all_batches_eccv18( test_set_2d, test_set_3d, training=False)

      total_err, joint_err, step_time, loss = evaluate_batches( sess, model,
        data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
        data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
        encoder_inputs, decoder_outputs)

      print("=============================\n"
            "Step-time (ms):      %.4f\n"
            "Val loss avg:        %.4f\n"
            "Val error avg (mm):  %.2f\n"
            "=============================" % ( 1000*step_time, loss, total_err ))

      for i in range(n_joints):
        # 6 spaces, right-aligned, 5 decimal places
        print("Error in joint {0:02d} (mm): {1:>5.2f}".format(i+1, joint_err[i]))
      print("=============================")

      # Log the error to tensorboard
      summaries = sess.run( model.err_mm_summary, {model.err_mm: total_err} )
      model.test_writer.add_summary( summaries, current_step )

      # Save the model
      print( "Saving the model... ", end="" )
      start_time = time.time()
      model.saver.save(sess, os.path.join(train_dir, 'checkpoint'), global_step=current_step )
      # model.saver.save(sess, os.path.join(train_dir, 'checkpoint'))
      print( "done in {0:.2f} ms".format(1000*(time.time() - start_time)) )

      print(train_dir)

      # Reset global time and loss
      step_time, loss = 0, 0

      sys.stdout.flush()


def get_action_subset( poses_set, action ):
  """
  Given a preloaded dictionary of poses, load the subset of a particular action

  Args
    poses_set: dictionary with keys k=(subject, action, seqname),
      values v=(nxd matrix of poses)
    action: string. The action that we want to filter out
  Returns
    poses_subset: dictionary with same structure as poses_set, but only with the
      specified action.
  """
  return {k:v for k, v in poses_set.items() if k[1] == action}



def evaluate_batches( sess, model,
  data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
  data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
  encoder_inputs, decoder_outputs, ):

  """
  Generic method that evaluates performance of a list of batches.
  May be used to evaluate all actions or a single action.

  Args
    sess
    model
    data_mean_3d
    data_std_3d
    dim_to_use_3d
    dim_to_ignore_3d
    data_mean_2d
    data_std_2d
    dim_to_use_2d
    dim_to_ignore_2d
    current_step
    encoder_inputs
    decoder_outputs
    current_epoch
  Returns

    total_err
    joint_err
    step_time
    loss
  """

  n_joints = 17 if not(FLAGS.predict_14) else 14
  nbatches = len( encoder_inputs )

  # Loop through test examples
  all_dists, start_time, loss = [], time.time(), 0.
  log_every_n_batches = 100
  for i in range(nbatches):


    enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
    dp = 1.0 # dropout keep probability is always 1 at test time
    step_loss, loss_summary, poses3d = model.step( sess, enc_in, dec_out, dp, isTraining=False )
    loss += step_loss

    # denormalize
    enc_in  = data_utils.unNormalizeData( enc_in,  data_mean_2d, data_std_2d, dim_to_ignore_2d )
    dec_out = data_utils.unNormalizeData( dec_out, data_mean_3d, data_std_3d, dim_to_ignore_3d )
    poses3d = data_utils.unNormalizeData( poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d )

    # Keep only the relevant dimensions
    dtu3d = np.hstack( (np.arange(3), dim_to_use_3d) ) if not(FLAGS.predict_14) else  dim_to_use_3d

    dec_out = dec_out[:, dtu3d]
    poses3d = poses3d[:, dtu3d]

    print (dec_out.shape[0], FLAGS.batch_size)
    print (poses3d.shape[0],FLAGS.batch_size)

    assert dec_out.shape[0] == FLAGS.batch_size
    assert poses3d.shape[0] == FLAGS.batch_size

    if FLAGS.procrustes:
      # Apply per-frame procrustes alignment if asked to do so
      for j in range(FLAGS.batch_size):
        gt  = np.reshape(dec_out[j,:],[-1,3])
        out = np.reshape(poses3d[j,:],[-1,3])
        _, Z, T, b, c = procrustes.compute_similarity_transform(gt,out,compute_optimal_scale=True)
        out = (b*out.dot(T))+c

        poses3d[j,:] = np.reshape(out,[-1,17*3] ) if not(FLAGS.predict_14) else np.reshape(out,[-1,14*3] )

    # Compute Euclidean distance error per joint
    sqerr = (poses3d - dec_out)**2 # Squared error between prediction and expected output
    dists = np.zeros( (sqerr.shape[0], n_joints) ) # Array with L2 error per joint in mm
    dist_idx = 0
    for k in np.arange(0, n_joints*3, 3):
      # Sum across X,Y, and Z dimenstions to obtain L2 distance
      dists[:,dist_idx] = np.sqrt( np.sum( sqerr[:, k:k+3], axis=1 ))
      dist_idx = dist_idx + 1

    all_dists.append(dists)
    assert sqerr.shape[0] == FLAGS.batch_size

  step_time = (time.time() - start_time) / nbatches
  loss      = loss / nbatches

  all_dists = np.vstack( all_dists )

  # Error per joint and total for all passed batches
  joint_err = np.mean( all_dists, axis=0 )
  total_err = np.mean( all_dists )

  return total_err, joint_err, step_time, loss


def eval_eccv18():
  """Get samples from a model and visualize them"""

  actions = data_utils.define_actions( FLAGS.action )

  # Load 3d data
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d  = data_utils.read_data_eccv18(
    FLAGS.data_dir, FLAGS.centering_2d, FLAGS.detector_2d, dim=3)


  train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_data_eccv18(
    FLAGS.data_dir, FLAGS.centering_2d, FLAGS.detector_2d, dim=2)


  device_count = {"GPU": 2} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto( device_count = device_count )) as sess:
    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    model = create_model(sess, actions, FLAGS.batch_size, for_eccv18=True)
    print("Model loaded")

    n_joints = 17 if not (FLAGS.predict_14) else 14
    encoder_inputs, decoder_outputs = model.get_all_batches_eccv18(test_set_2d, test_set_3d, training=False)

    total_err, joint_err, step_time, loss = evaluate_batches(sess, model,
                                                             data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
                                                             data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
                                                             encoder_inputs, decoder_outputs)

    print("=============================\n"
          "Step-time (ms):      %.4f\n"
          "Val loss avg:        %.4f\n"
          "Val error avg (mm):  %.2f\n"
          "=============================" % (1000 * step_time, loss, total_err))

    for i in range(n_joints):
    # 6 spaces, right-aligned, 5 decimal places
      print("Error in joint {0:02d} (mm): {1:>5.2f}".format(i + 1, joint_err[i]))
    print("=============================")



def generate_3dpose_eccv18():
  """Get samples from a model and visualize them"""

  actions = data_utils.define_actions( FLAGS.action )

  # Load 3d & 2d data
  _, _, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d  = data_utils.read_data_eccv18(
    FLAGS.data_dir, FLAGS.centering_2d, FLAGS.detector_2d, dim=3, for_submission = True)

  train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_data_eccv18(
    FLAGS.data_dir, FLAGS.centering_2d, FLAGS.detector_2d, dim=2, for_submission = True)


  # Load test filename_list (Unshuffled)
  file_list = []
  split_path = os.path.join(FLAGS.data_dir, "split", 'Test_list.csv')
  with open(split_path, 'r') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
      file_list.append(row[0].split('.jp')[0])

  device_count = {"GPU": 2} if FLAGS.use_cpu else {"GPU": 1}
  idx_file =0
  with tf.Session(config=tf.ConfigProto( device_count = device_count )) as sess:
    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    model = create_model(sess, actions, FLAGS.batch_size, for_eccv18=True)
    print("Model loaded")

    n_joints = 17 if not (FLAGS.predict_14) else 14
    encoder_inputs = model.get_all_batches_2D_eccv18(test_set_2d)
    nbatches = len(encoder_inputs)

    for i in range(nbatches):

      enc_in = encoder_inputs[i]
      dp = 1.0  # dropout keep probability is always 1 at test time
      poses3d = model.step_only_enc(sess, enc_in, dp, isTraining=False)

      # denormalize
      poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)

      # Keep only the relevant dimensions
      dtu3d = np.hstack((np.arange(3), dim_to_use_3d)) if not (FLAGS.predict_14) else dim_to_use_3d
      poses3d = poses3d[:, dtu3d]

      batch_size = poses3d.shape[0]
      n_joints = 17 if not(FLAGS.predict_14) else 14

      for i in range(batch_size):
        pose3d_sample = poses3d[i].reshape(n_joints, -1)
        np.savetxt(FLAGS.prediction_dir + file_list[idx_file] + ".csv", pose3d_sample, delimiter=",", fmt='%.3f')
        idx_file +=1



def main(_):
  if FLAGS.mode == 'train':
    train_eccv18()
  elif FLAGS.mode == 'eval':
    eval_eccv18()
  elif FLAGS.mode == 'generate':
    generate_3dpose_eccv18()


if __name__ == "__main__":
  tf.app.run()
