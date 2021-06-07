from UQpy.Distributions import Lognormal, Uniform, Normal
dist1 = Lognormal(s=0.19804, scale=98058.0675, loc=0.0) 

from UQpy.SampleMethods import LHS, MCS
training_sampling = LHS(dist_object=dist1, nsamples=20)
training_samples = training_sampling.samples.reshape(20,1)

validation_sampling = LHS(dist_object=dist1, nsamples=30)
validation_samples = validation_sampling.samples.reshape(30,1)

from UQpy.RunModel import RunModel

model_serial_third_party=RunModel(model_script='PythonAsThirdParty_model.py', model_object_name='run_model',
    input_template='elastic_contact_sphere_1D.py', var_names=['k'],
    output_script='process_3rd_party_output.py')

model_serial_third_party.run(samples=training_samples)
qoi = model_serial_third_party.qoi_list

maximum_displacement=list(map(abs, qoi[:20]))

model_serial_third_party_validation=RunModel(model_script='PythonAsThirdParty_model.py', model_object_name='run_model',
    input_template='elastic_contact_sphere_1D.py', var_names=['k'],
    output_script='process_3rd_party_output.py')

model_serial_third_party_validation.run(samples=validation_samples)
maximum_displacement_validation=list(map(abs, qoi[:30]))

from UQpy.Surrogates import Kriging
K = Kriging(reg_model='Linear', corr_model='Gaussian', nopt=20, corr_model_params=[1])
K.fit(samples=training_samples, values=maximum_displacement)
prediction_sampling=MCS(dist_object=[dist1], nsamples=1000,  verbose=True)
prediction_results=K.predict(prediction_sampling.samples)

import statistics
print(statistics.variance(prediction_results))

a=1