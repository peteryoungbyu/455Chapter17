using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using _455chapter17.API.Data;
using _455chapter17.API.Models;

namespace _455chapter17.API.Controllers;

[ApiController]
[Route("api/orders")]
public class OrdersController(AppDbContext db) : ControllerBase
{
    [HttpGet]
    public async Task<IActionResult> GetAll()
    {
        var orders = await db.Orders
            .Include(o => o.Customer)
            .OrderByDescending(o => o.OrderDatetime)
            .Take(500)
            .Select(o => new
            {
                o.OrderId,
                o.CustomerId,
                CustomerName = o.Customer != null ? o.Customer.FullName : "",
                OrderDatetime = o.OrderDatetime,
                o.PaymentMethod,
                o.DeviceType,
                o.OrderTotal,
                RiskScore = NormalizeProbability(o.RiskScore),
                o.IsFraud
            })
            .ToListAsync();

        return Ok(orders);
    }

    [HttpPost]
    public async Task<IActionResult> PlaceOrder([FromBody] PlaceOrderPayload payload)
    {
        var productIds = payload.Items.Select(i => i.ProductId).ToList();
        var products = await db.Products
            .Where(p => productIds.Contains(p.ProductId))
            .ToDictionaryAsync(p => p.ProductId);

        decimal subtotal = 0;
        var items = new List<OrderItem>();
        foreach (var item in payload.Items)
        {
            if (!products.TryGetValue(item.ProductId, out var product))
                return BadRequest($"Product {item.ProductId} not found.");

            var lineTotal = product.Price * item.Quantity;
            subtotal += lineTotal;
            items.Add(new OrderItem
            {
                ProductId = item.ProductId,
                Quantity = item.Quantity,
                UnitPrice = product.Price,
                LineTotal = lineTotal
            });
        }

        const decimal shippingFee = 8.99m;
        var tax = Math.Round(subtotal * 0.08m, 2);
        var total = subtotal + shippingFee + tax;

        var order = new Order
        {
            CustomerId = payload.CustomerId,
            OrderDatetime = DateTime.UtcNow,
            BillingZip = payload.BillingZip,
            ShippingZip = payload.ShippingZip,
            PaymentMethod = payload.PaymentMethod,
            DeviceType = payload.DeviceType,
            IpCountry = "US",
            PromoUsed = !string.IsNullOrEmpty(payload.PromoCode),
            PromoCode = payload.PromoCode,
            OrderSubtotal = subtotal,
            ShippingFee = shippingFee,
            TaxAmount = tax,
            OrderTotal = total,
            RiskScore = 50.0m,
            IsFraud = false,
            OrderItems = items
        };

        db.Orders.Add(order);
        await db.SaveChangesAsync();

        return Ok(new
        {
            order.OrderId,
            order.CustomerId,
            OrderDatetime = order.OrderDatetime,
            order.PaymentMethod,
            order.DeviceType,
            order.OrderTotal,
            RiskScore = NormalizeProbability(order.RiskScore),
            order.IsFraud
        });
    }

    private static decimal NormalizeProbability(decimal rawValue)
    {
        var normalized = rawValue > 1m ? rawValue / 100m : rawValue;
        return Math.Round(Math.Clamp(normalized, 0m, 1m), 4);
    }
}

public record PlaceOrderPayload(
    int CustomerId,
    string PaymentMethod,
    string DeviceType,
    string BillingZip,
    string ShippingZip,
    string? PromoCode,
    List<OrderItemRequest> Items
);

public record OrderItemRequest(int ProductId, int Quantity);
