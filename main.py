from utils import parser
from utils import processor


def main():
    args = parser.Parser().args

    process = processor.Processor(args)

    process.start()


if __name__ == '__main__':
    main()
