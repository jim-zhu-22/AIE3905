"""Freeze the installed API dependency closure for Python 3.12 Windows/Linux."""

from importlib.metadata import distribution
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    root = Path(__file__).resolve().parents[1]
    direct = root / "api_service/requirements.txt"
    pending = [Requirement(line).name for line in direct.read_text().splitlines() if line.strip()]
    versions = {}
    environments = []
    for system, platform in (("Windows", "win32"), ("Linux", "linux")):
        environment = default_environment()
        environment.update(platform_system=system, sys_platform=platform, extra="")
        environments.append(environment)
    while pending:
        name = canonicalize_name(pending.pop())
        if name in versions:
            continue
        package = distribution(name)
        versions[name] = package.version
        for dependency in package.requires or []:
            requirement = Requirement(dependency)
            if requirement.marker is None or any(requirement.marker.evaluate(env) for env in environments):
                pending.append(requirement.name)
    lines = ["# API runtime closure from the validated Python 3.12 environment.",
             "# Recreate using the same Python minor version; real-model dependencies are separate."]
    lines += [f"{name}=={versions[name]}" for name in sorted(versions)]
    target = root / "api_service/requirements.lock"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Locked {len(versions)} API runtime packages in {target}")


if __name__ == "__main__":
    main()
