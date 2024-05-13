import zmq
from multiprocessing import Pool
import os


def process_data(klines):
    # Здесь ваш код обработки данных
    print(f"Processing data: {klines}")


if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")  # адрес сервера ZeroMQ

    # Создаём пул процессов один раз
    pool = Pool(os.cpu_count())

    try:
        while True:
            klines = socket.recv_json()
            pool.apply_async(process_data, (klines,))
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        pool.close()  # Закрытие пула
        pool.join()  # Дождаться завершения всех процессов
        socket.close()  # Закрытие сокета
        context.term()  # Закрытие контекста ZeroMQ
