# the LDCast repo needs to be cloned in this same directory

from ldcast.ldcast import forecast
from ldcast.ldcast.models.diffusion.plms import PLMSSampler

def ldcast_forecast(R, num_diffusion_iters = 50, ensemble_members = 1, sampler_progbar = False):
    ldm_weights_fn = 'ldcast/models/genforecast/genforecast-radaronly-256x256-20step.pt'
    autoenc_weights_fn = 'ldcast/models/autoenc/autoenc-32-0.01.pt'

    fc = forecast.ForecastDistributed(
        ldm_weights_fn=ldm_weights_fn, autoenc_weights_fn=autoenc_weights_fn, sampler_progbar = sampler_progbar
    )
    if len(R.shape) > 4 or len(R.shape) < 3:
        raise ValueError(f'The precipitation field R should have 4 dimensions, and it has {len(R.shape)}')
    if len(R.shape) == 3:
        print('R has 3 dimensions, so extending on the 0th axis')
        R = R[None]
    
    pred = fc(R,
              ensemble_members = ensemble_members,
              num_diffusion_iters = num_diffusion_iters).swapaxes(3, 4).swapaxes(2, 3).squeeze()

    return pred