using Microsoft.EntityFrameworkCore;
using Npgsql;
using _455chapter17.API.Data;

var builder = WebApplication.CreateBuilder(args);

LoadEnvironmentVariables(builder.Environment.ContentRootPath);

var port = Environment.GetEnvironmentVariable("PORT") ?? "5000";
builder.WebHost.UseUrls($"http://0.0.0.0:{port}");

var connectionString = Environment.GetEnvironmentVariable("SUPABASE_DB_URL")
    ?? builder.Configuration.GetConnectionString("Default")
    ?? throw new InvalidOperationException("No database connection string configured. Set SUPABASE_DB_URL or ConnectionStrings:Default.");

connectionString = NormalizePostgresConnectionString(connectionString);

builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseNpgsql(connectionString));

builder.Services.AddControllers()
    .AddJsonOptions(options =>
        options.JsonSerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.SnakeCaseLower);
builder.Services.AddCors(options =>
{
    options.AddPolicy("FrontendDev", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyHeader()
              .AllowAnyMethod();
    });
});
builder.Services.AddOpenApi();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseCors("FrontendDev");

app.UseAuthorization();

app.MapControllers();

app.Run();

static void LoadEnvironmentVariables(string contentRootPath)
{
    // Attempt to load .env from the API folder, backend folder, or repository root.
    var candidates = new[]
    {
        Path.Combine(contentRootPath, ".env"),
        Path.Combine(contentRootPath, "..", ".env"),
        Path.Combine(contentRootPath, "..", "..", ".env")
    };

    foreach (var candidate in candidates)
    {
        var fullPath = Path.GetFullPath(candidate);
        if (!File.Exists(fullPath))
        {
            continue;
        }

        foreach (var rawLine in File.ReadAllLines(fullPath))
        {
            var line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith("#"))
            {
                continue;
            }

            var separator = line.IndexOf('=');
            if (separator <= 0)
            {
                continue;
            }

            var key = line[..separator].Trim();
            var value = line[(separator + 1)..].Trim().Trim('"', '\'');

            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable(key)))
            {
                Environment.SetEnvironmentVariable(key, value);
            }
        }

        break;
    }
}

static string NormalizePostgresConnectionString(string rawConnectionString)
{
    var trimmed = rawConnectionString.Trim();
    if (!trimmed.StartsWith("postgres://", StringComparison.OrdinalIgnoreCase)
        && !trimmed.StartsWith("postgresql://", StringComparison.OrdinalIgnoreCase))
    {
        return trimmed;
    }

    var uri = new Uri(trimmed);
    var userInfoParts = uri.UserInfo.Split(':', 2);
    var username = userInfoParts.Length > 0 ? Uri.UnescapeDataString(userInfoParts[0]) : string.Empty;
    var password = userInfoParts.Length > 1 ? Uri.UnescapeDataString(userInfoParts[1]) : string.Empty;
    var database = uri.AbsolutePath.Trim('/');

    var builder = new NpgsqlConnectionStringBuilder
    {
        Host = uri.Host,
        Port = uri.IsDefaultPort ? 5432 : uri.Port,
        Username = username,
        Password = password,
        Database = string.IsNullOrWhiteSpace(database) ? "postgres" : database
    };

    var query = uri.Query.TrimStart('?');
    if (!string.IsNullOrWhiteSpace(query))
    {
        foreach (var pair in query.Split('&', StringSplitOptions.RemoveEmptyEntries))
        {
            var kv = pair.Split('=', 2);
            var key = kv[0].Trim();
            var value = kv.Length > 1 ? Uri.UnescapeDataString(kv[1].Trim()) : string.Empty;

            if (key.Equals("sslmode", StringComparison.OrdinalIgnoreCase)
                && Enum.TryParse<SslMode>(value, true, out var sslMode))
            {
                builder.SslMode = sslMode;
            }
        }
    }

    return builder.ConnectionString;
}
