# from UQpy.RunModel import RunModel
# from UQpy.Distributions import Uniform
# from UQpy.SampleMethods import MCS
# import numpy as np
# import matplotlib.pyplot as plt
# from UQpy.Surrogates import *
# from UQpy.SampleMethods import MCS
# from UQpy.Distributions import Uniform

# distribution=Uniform(0.5, 3.0)

# training_sampling = MCS(dist_object=distribution, nsamples=30)
# training_samples=training_sampling.samples

# validation_sampling = MCS(dist_object=distribution, nsamples=20)
# validation_samples = validation_sampling.samples

# boucwen = RunModel(model_script='model_1D.py', model_object_name='boucwen_runmodel', var_names=['r0'])
# boucwen.run(samples=training_samples)

# maximum_displacement = boucwen.qoi_list[:30]

# boucwen.run(samples=validation_samples)

# maximum_displacement_validation=boucwen.qoi_list[-20:]

# from UQpy.Surrogates import PCE, PolyChaosLstsq, Polynomials

# polys = Polynomials(dist_object=distribution, degree=1)
# lstsq = PolyChaosLstsq(poly_object=polys)
# pce = PCE(method=lstsq)

# pce.fit(training_samples,np.array(maximum_displacement).reshape(30,1))

# prediction_sampling=MCS(dist_object=[distribution], nsamples=100,  verbose=True)
# prediction_results=pce.predict(prediction_sampling.samples)

# from UQpy.Surrogates import ErrorEstimation
# error = ErrorEstimation(surr_object=pce)
# print('Error from least squares regression is: ', error.validation(validation_sampling.samples, np.array(maximum_displacement_validation)))


from UQpy.RunModel import RunModel
from UQpy.Distributions import Uniform, Normal, JointInd
from UQpy.SampleMethods import MCS
import numpy as np
import matplotlib.pyplot as plt
from UQpy.Surrogates import *
from UQpy.SampleMethods import MCS
distribution1=Normal(1e5, 2*1e4)
distribution2=Uniform(0.01, 0.89)


training_sampling = MCS(dist_object=JointInd(marginals=[distribution1, distribution2]), nsamples=20)
training_samples=training_sampling.samples

model_serial_third_party=RunModel(samples=training_samples,  model_script='PythonAsThirdParty_model_2D.py',
    input_template='elastic_contact_sphere.py', var_names=['k', 'f0'],
    output_script='process_3rd_party_output.py', model_object_name='read_output')

maximum_displacement = model_serial_third_party.qoi_list[:20]

from UQpy.Surrogates import PCE, PolyChaosRidge, Polynomials

polys = Polynomials(dist_object=JointInd(marginals=[distribution1, distribution2]), degree=2)
lstsq = PolyChaosRidge(poly_object=polys)
pce = PCE(method=lstsq)

pce.fit(training_samples,np.array(maximum_displacement).reshape(20,1))

a=1