import kfp
from kfp import components
from kfp import dsl
from kfp import onprem


def train_op(epochs, validations, workers, trainset, input, filenames, target,train_size, learn_rate):
  return dsl.ContainerOp(
    name='Train',
    image='pascalschroeder/ml-train-test',
    arguments=[
      '--epochs', epochs,
      '--validations', validations,
      '--workers', workers,
      '--trainset', trainset,
      '--input', input,
      '--filenames', filenames,
      '--target', target,
      '--train_size', train_size,
      '--learn_rate', learn_rate
    ],
    file_outputs={
        'model': '/home/rootkrause/Documents/tensorflow_test/model/model.h5'
    }
  )

def load_op(workers, testset, input, filenames, target, model, output):
  return dsl.ContainerOp(
    name='Load',
    image='pascalschroeder/ml-load-test',
    arguments=[
      '--workers', workers,
      '--testset', testset,
      '--input', input,
      '--filenames', filenames,
      '--target', target,
      '--model', model,
      '--output', output
    ],
    file_outputs={
        'result': './result.txt'
    }
  )


@dsl.pipeline(
  name='ML Test Pipeline',
  description='Test'
)
def train_pipeline(output, pvc_name="train-vol", pvc_path="/mnt", epochs=30, validations=10, trainset='./cut', testset='./cut', input='./train.csv', filenames='id', target='has_scratch', train_size=0.8, learn_rate=0.0001, workers=2):

  train = train_op(epochs, validations, workers, trainset, input, filenames, target, train_size, learn_rate).apply(onprem.mount_pvc("train-vol", 'local-storage', "/mnt"))
  load = load_op(workers, testset, input, filenames, target, train.outputs['model'], output).apply(onprem.mount_pvc("train-vol", 'local-storage', "/mnt"))

if __name__ == '__main__':
  kfp.compiler.Compiler().compile(train_pipeline, __file__ + '.tar.gz')