//! Dependency-free subprocess bridge. Run from Compactification-Theory root.
use std::env;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: compile_exam ADAPTER_JSON NEW_PACKET_JSON");
        return ExitCode::from(2);
    }
    let python = env::var("DISTILL_PYTHON").unwrap_or_else(|_| "python".into());
    match Command::new(python)
        .args(["-m", "Distill", "compile", "--adapter", &args[1], "--out", &args[2]])
        .status()
    {
        Ok(status) if status.success() => ExitCode::SUCCESS,
        Ok(status) => {
            eprintln!("Distill compile exited: {status}");
            ExitCode::FAILURE
        }
        Err(err) => {
            eprintln!("Distill launch failed: {err}");
            ExitCode::FAILURE
        }
    }
}
