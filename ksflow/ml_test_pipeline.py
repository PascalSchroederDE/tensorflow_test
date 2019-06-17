import kfp
from kfp import components
from kfp import dsl
from kfp import onprem


def train_op(epochs, validations, workers, pvc_path, trainset, input, filenames, target,train_size, learn_rate, output):
  return dsl.ContainerOp(
    name='Train',
    image='pascalschroeder/ml-train-test',
    arguments=[
      '--epochs', epochs,
      '--validations', validations,
      '--workers', workers,
      '--pvc_path', pvc_path,
      '--trainset', trainset,
      '--input', input,
      '--filenames', filenames,
      '--target', target,
      '--train_size', train_size,
      '--learn_rate', learn_rate,
      '--output', output
    ],
    file_outputs={
        'output': '/output.txt'
     } 
  )

def load_op(workers, pvc_path, testset, input, filenames, target, model, result):
  return dsl.ContainerOp(
    name='Load',
    image='pascalschroeder/ml-load-test',
    arguments=[
      '--workers', workers,
      '--pvc_path', pvc_path,
      '--testset', testset,
      '--input', input,
      '--filenames', filenames,
      '--target', target,
      '--model', model,
      '--output', result
    ],
  )


@dsl.pipeline(
  name='ML Test Pipeline',
  description='Test'
)
def train_pipeline(output="/mnt/model.h5", result="/mnt/results.txt", pvc_name="train-vol", pvc_path="/mnt", epochs=30, validations=10, trainset='/cut', testset='/cut', input='/train.csv', filenames='id', target='has_scratch', train_size=0.8, learn_rate=0.0001, workers=2):
  train = train_op(epochs, validations, workers,  pvc_path, trainset, input, filenames, target, train_size, learn_rate, output).apply(onprem.mount_pvc("train-vol", 'local-storage', "/mnt"))
  load = load_op(workers, pvc_path, testset, input, filenames, target, train.outputs['output'], result).apply(onprem.mount_pvc("train-vol", 'local-storage', "/mnt"))

if __name__ == '__main__':
  kfp.compiler.Compiler().compile(train_pipeline, __file__ + '.tar.gz')