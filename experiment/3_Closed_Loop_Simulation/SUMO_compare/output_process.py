import xml.dom.minidom


def process_vel_dis():
    # Experiment Indicators
    avg_flow_vel = []
    avg_space_headway = []

    dom = xml.dom.minidom.parse('roadgraph.output.xml')
    root = dom.documentElement
    lanes = root.getElementsByTagName("lane")
    for lane in lanes:
        vehicles = lane.getElementsByTagName("vehicle")
        for ego_veh in vehicles:
            avg_flow_vel.append(float(ego_veh.getAttribute('speed')))
            min_space_headway = 100
            for other_veh in vehicles:
                if ego_veh.getAttribute('id') == other_veh.getAttribute('id'):
                    continue
                if other_veh.getAttribute('pos') > ego_veh.getAttribute('pos'):
                    min_space_headway = min(min_space_headway,
                                            float(other_veh.getAttribute('pos')) - float(ego_veh.getAttribute('pos')))
            if min_space_headway < 100:
                avg_space_headway.append(min_space_headway)

    print("avg_flow_vel: ", sum(avg_flow_vel) / len(avg_flow_vel))
    print("avg_space_headway: ", sum(avg_space_headway) / len(avg_space_headway))


def process_travel_time():
    # Experiment Indicators
    travel_duration = []

    dom = xml.dom.minidom.parse('roadgraph.output.xml')
    root = dom.documentElement
    vehicles = root.getElementsByTagName("vehicle")
    for veh in vehicles:
        depart_time = float(veh.getAttribute('depart'))
        arrival_time = float(veh.getAttribute('arrival'))
        travel_duration.append(arrival_time - depart_time)
    print("avg_travel_duration: ", sum(travel_duration) / len(travel_duration))


def main():
    process_vel_dis()
    # process_travel_time()


if __name__ == "__main__":
    main()
