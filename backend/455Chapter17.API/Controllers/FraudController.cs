using Microsoft.AspNetCore.Mvc;

namespace _455Chapter17.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class FraudController : ControllerBase
{
	[HttpGet("ping")]
	public IActionResult Ping()
	{
		return Ok(new
		{
			message = "Backend connection successful",
			timestampUtc = DateTime.UtcNow
		});
	}
}
