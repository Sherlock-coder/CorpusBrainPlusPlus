import jsonlines

if __name__ == "__main__":
    with open('./dev.source', 'r') as source:
        with open('./dev.target', 'r') as target:
            with jsonlines.open('./real_generated_data.jsonl.dev', 'w') as writer:
                slst = [i.strip() for i in source.readlines()]
                tlst = [i.strip() for i in target.readlines()]
                for s, t in zip(slst, tlst):
                    writer.write({'source': s, 'target': t})