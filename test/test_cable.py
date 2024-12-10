from bridgezoo.cablebridge_di.envs.cablebridge_models import CableFixStrands

c = CableFixStrands(num_strands=10, stress_init=0, stress_delta=10)
for i in range(19):
    a = c.action_space.sample()
    print(a)
