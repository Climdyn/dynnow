from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.nowcasts.steps import forecast

def STEPS_forecast(R, ensemble_members = 1, STEPS_kwargs = {}):

    motion = dense_lucaskanade(R)
    pred = forecast(R, motion, 20, n_ens_members = ensemble_members, precip_thr = -10, timestep = 5, kmperpixel = 1, **STEPS_kwargs)

    return pred.swapaxes(0, 1)

    