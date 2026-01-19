import argparse
from src.dl.predictor import predict

def main():
    parser = argparse.ArgumentParser(description="CLI Sentiment Predictor (GRU)")
    parser.add_argument("text", type=str, help="Text to classify")
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = predict(
        text=args.text,
        max_len=args.max_len,
        threshold=args.threshold
    )

    print(f"Sentiment: {result['sentiment']}")
    print(f"Probability(positive): {result['prob_positive']:.4f}")

if __name__ == "__main__":
    main()
