import random

routes = {
    'r1': 'E0 E1',
    'r2': 'E2 E1'
}


def genRou(period: float, keep_lane_rate: float, turn_right_rate: float) -> None:
    with open('roadgraph.rou.xml', 'w') as rf:
        print('''<routes>''', file=rf)
        for i in range(int(100/period)):
            routeRand = random.random()
            if routeRand <= 3/4:
                route = 'r1'
                departLane = random.choice([0, 1, 2])
                klRandom = random.random()
                if departLane == 0:
                    if klRandom <= keep_lane_rate:
                        arrivalLane = 0
                    else:
                        arrivalLane = 1
                elif departLane == 1:
                    if klRandom <= keep_lane_rate:
                        arrivalLane = 1
                    else:
                        trRandom = random.random()
                        if trRandom <= turn_right_rate:
                            arrivalLane = 0
                        else:
                            arrivalLane = 2
                else:
                    if klRandom <= keep_lane_rate:
                        arrivalLane = 2
                    else:
                        arrivalLane = 1
            else:
                route = 'r2'
                departLane = 0
                arrivalLane = 0
            print(
                '''    <vehicle id="{}" depart="{}" departSpeed="{}" departLane="{}" arrivalLane="{}">'''.format(
                    str(i), str(i*period), 5, departLane, arrivalLane
                ),
                file=rf
            )
            print(
                '''        <route edges="{}"/>'''.format(routes[route]),
                file=rf
            )
            print('''    </vehicle>''', file=rf)
        print('''</routes>''', file=rf)


if __name__ == '__main__':
    genRou(2.0, 0.8, 0.2)
