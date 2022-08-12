
def main():
    log_path = 'results/HIO/stdout.log'

    with open(log_path, 'r') as file:
        lines = file.readlines()

    times = []
    for line in lines:
        line = line.strip()
        if 'Average time' in line:
            time = float(line.split(":")[-1])
            times.append(time)

    average_time = sum(times) / len(times)
    print(f"Average time: {average_time}")

if __name__ == '__main__':
    main()
