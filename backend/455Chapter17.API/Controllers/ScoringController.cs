using System.Diagnostics;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using _455chapter17.API.Data;
using _455chapter17.API.Models;

namespace _455chapter17.API.Controllers;

[ApiController]
[Route("api/scoring")]
public class ScoringController(AppDbContext db, IWebHostEnvironment env, ILogger<ScoringController> logger) : ControllerBase
{
    [HttpPost("run")]
    public async Task<IActionResult> RunScoring()
    {
        // Scripts and model are published alongside the app via .csproj Content items.
        // In local dev they live two levels up at the repo root.
        var contentRoot = env.ContentRootPath;
        var repoRoot = Path.GetFullPath(Path.Combine(contentRoot, "..", ".."));

        var scriptPath = Path.Combine(contentRoot, "scripts", "run_fraud_scoring.py");
        if (!System.IO.File.Exists(scriptPath))
            scriptPath = Path.Combine(repoRoot, "scripts", "run_fraud_scoring.py");

        var modelPath = Environment.GetEnvironmentVariable("FRAUD_MODEL_PATH")
            ?? Path.Combine(contentRoot, "crispdm-pipeline-model", "fraud_model.sav");
        if (!System.IO.File.Exists(modelPath))
            modelPath = Path.Combine(repoRoot, "crispdm-pipeline-model", "fraud_model.sav");
        var pythonExecutable = Environment.GetEnvironmentVariable("PYTHON_EXECUTABLE");
        var pythonVersionSelector = Environment.GetEnvironmentVariable("PYTHON_VERSION_SELECTOR");

        if (string.IsNullOrWhiteSpace(pythonExecutable))
        {
            // Prefer the bundled venv created during Railway build
            var venvPython = Path.Combine(contentRoot, "venv", "bin", "python");
            if (System.IO.File.Exists(venvPython))
            {
                pythonExecutable = venvPython;
            }
            else if (OperatingSystem.IsWindows())
            {
                pythonExecutable = "py";
                pythonVersionSelector ??= "-3.12";
            }
            else
            {
                pythonExecutable = "python3";
            }
        }

        if (!System.IO.File.Exists(scriptPath))
        {
            logger.LogError("Scoring script not found at {ScriptPath}", scriptPath);
            return StatusCode(500, new { message = "Scoring script not found." });
        }

        if (!System.IO.File.Exists(modelPath))
        {
            logger.LogError("Model file not found at {ModelPath}", modelPath);
            return StatusCode(500, new { message = "Model file not found." });
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = pythonExecutable,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = contentRoot
        };
        if (!string.IsNullOrWhiteSpace(pythonVersionSelector))
        {
            startInfo.ArgumentList.Add(pythonVersionSelector);
        }
        startInfo.ArgumentList.Add(scriptPath);
        startInfo.ArgumentList.Add("--model-path");
        startInfo.ArgumentList.Add(modelPath);

        using var process = Process.Start(startInfo);
        if (process is null)
        {
            logger.LogError("Failed to start scoring process using executable {PythonExecutable}", pythonExecutable);
            return StatusCode(500, new { message = "Failed to start scoring process." });
        }

        var stdoutTask = process.StandardOutput.ReadToEndAsync();
        var stderrTask = process.StandardError.ReadToEndAsync();

        await process.WaitForExitAsync(HttpContext.RequestAborted);

        var stdout = await stdoutTask;
        var stderr = await stderrTask;

        if (process.ExitCode != 0)
        {
            logger.LogError("Scoring process failed with exit code {ExitCode}. stderr: {Stderr}", process.ExitCode, stderr);
            return StatusCode(500, new
            {
                message = "Scoring process failed.",
                stderr,
                stdout
            });
        }

        var scored = ParseScoredCount(stdout);

        return Ok(new { message = "Scoring complete.", scored });
    }

    [HttpGet("queue")]
    public async Task<IActionResult> GetQueue()
    {
        var queue = await db.DeliveryScores
            .Include(ds => ds.Order)
            .ThenInclude(o => o!.Customer)
            .OrderByDescending(ds => ds.LateDeliveryProbability)
            .Take(100)
            .Select(ds => new
            {
                OrderId = ds.OrderId,
                CustomerName = ds.Order != null && ds.Order.Customer != null
                    ? ds.Order.Customer.FullName : "",
                OrderTotal = ds.Order != null ? ds.Order.OrderTotal : 0,
                RiskScore = ds.Order != null ? NormalizeProbability(ds.Order.RiskScore) : 0,
                FraudProbability = NormalizeProbability(ds.LateDeliveryProbability),
                ScoredAt = ds.ScoredAt
            })
            .ToListAsync();

        return Ok(queue);
    }

    private static decimal NormalizeProbability(decimal rawValue)
    {
        // Legacy source data stores risk scores on a 0-100 scale, while UI expects 0-1.
        var normalized = rawValue > 1m ? rawValue / 100m : rawValue;
        var clamped = Math.Clamp(normalized, 0m, 1m);
        return Math.Round(clamped, 4);
    }

    private static int ParseScoredCount(string stdout)
    {
        if (string.IsNullOrWhiteSpace(stdout))
        {
            return 0;
        }

        var lines = stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        for (var index = lines.Length - 1; index >= 0; index--)
        {
            try
            {
                using var json = JsonDocument.Parse(lines[index]);
                if (json.RootElement.TryGetProperty("scored", out var scoredElement)
                    && scoredElement.ValueKind == JsonValueKind.Number
                    && scoredElement.TryGetInt32(out var scored))
                {
                    return scored;
                }
            }
            catch (JsonException)
            {
                // Ignore non-JSON lines and continue scanning.
            }
        }

        return 0;
    }
}
