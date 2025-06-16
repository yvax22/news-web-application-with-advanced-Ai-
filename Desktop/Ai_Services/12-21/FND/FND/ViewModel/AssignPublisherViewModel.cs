using Microsoft.AspNetCore.Mvc.Rendering;
using System.Collections.Generic;

namespace FND.ViewModel
{
    public class AssignPublisherViewModel
    {
        public string SelectedUserId { get; set; }

        public List<SelectListItem> Users { get; set; } = new List<SelectListItem>();
    }
}
