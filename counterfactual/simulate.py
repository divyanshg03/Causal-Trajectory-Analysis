def increase_speed(traj, factor=1.5):
    new_traj = []

    for i in range(len(traj)):
        if i == 0:
            new_traj.append(traj[i])
        else:
            dx = traj[i][0] - traj[i-1][0]
            dy = traj[i][1] - traj[i-1][1]

            dx *= factor
            dy *= factor

            new_x = new_traj[i-1][0] + dx
            new_y = new_traj[i-1][1] + dy

            new_traj.append((new_x, new_y))

    return new_traj