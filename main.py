

def main():

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    now = datetime.strptime(now, '%Y-%m-%d %H:%M:%S')
    trade = StartTrade()
    trade.start()

if __name__ == '__main__':
    main()