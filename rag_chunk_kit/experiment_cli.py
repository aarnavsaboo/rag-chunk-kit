from argparse import ArgumentParser
from pathlib import Path
import json

from .experiment_report import summarize
from .experiments import Experiment,expand,run


def read_plan(path:str)->list[Experiment]:
    return [Experiment(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def read_rows(path:str)->list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser=ArgumentParser()
    sub=parser.add_subparsers(dest="cmd",required=True)

    plan=sub.add_parser("plan")
    plan.add_argument("config")

    execute=sub.add_parser("run")
    execute.add_argument("plan")
    execute.add_argument("--out",required=True)

    report=sub.add_parser("report")
    report.add_argument("path")

    args=parser.parse_args()
    if args.cmd=="plan":
        config=json.loads(Path(args.config).read_text())
        for item in expand(config):
            print(json.dumps(item.to_dict(),sort_keys=True))
    elif args.cmd=="run":
        rows=[run(item) for item in read_plan(args.plan)]
        target=Path(args.out)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows),encoding="utf-8")
        print(json.dumps({"experiments":len(rows)}))
    else:
        print(json.dumps(summarize(read_rows(args.path)),indent=2))


if __name__=="__main__":
    main()
