import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    required property var row
    property bool selected: false
    signal clicked()

    height: 150
    radius: 10
    color: selected ? "#123052" : (row.rank <= 3 ? "#13263A" : (hovered ? "#111C2D" : "#0E1623"))
    border.color: selected ? "#36A3FF" : (row.rank <= 3 ? "#E6B85C" : "#1B2A41")
    property bool hovered: false

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 4
        RowLayout {
            Layout.fillWidth: true
            Label { text: row.id + " · " + row.name; color: "#E3EEFF"; font.bold: true; Layout.fillWidth: true; elide: Text.ElideRight }
            Label { text: "#" + (row.rank || "-"); color: row.rank <= 3 ? "#FFD27A" : "#8FCBFF"; font.pixelSize: 11; font.bold: row.rank <= 3 }
            Label { text: "Score " + (row.score || 0); color: "#CBE5FF"; font.pixelSize: 11; font.bold: true }
            Label { text: row.status; color: "#8FCBFF"; font.pixelSize: 11 }
        }
        Label { text: row.family + " | G" + row.generation + " | " + row.origin; color: "#9FB5D7"; font.pixelSize: 12 }
        Label { text: "Regime: " + (row.regime || "trend-following"); color: "#F5CF88"; font.pixelSize: 11 }
        RowLayout {
            Label { text: "Fit " + row.fitness; color: "#D3E7FF"; font.pixelSize: 12 }
            Label { text: "Rob " + row.robustness; color: "#8CE0C9"; font.pixelSize: 12 }
            Label { text: "Val " + row.validation; color: "#EACB95"; font.pixelSize: 12 }
        }
        RowLayout {
            Label { text: "Trades " + (row.trade_count || 0); color: "#C8D8EE"; font.pixelSize: 11 }
            Label { text: "Win " + (row.win_rate || 0) + "%"; color: "#9CD7B7"; font.pixelSize: 11 }
            Label { text: "PnL " + (row.pnl || 0) + "%"; color: "#A6C9FF"; font.pixelSize: 11 }
            Label { text: "DD " + (row.drawdown || 0) + "%"; color: "#F2B7B7"; font.pixelSize: 11 }
        }
        RowLayout {
            Label { text: "AvgTrade " + (row.avg_trade_return || 0) + "%"; color: "#CDE4FF"; font.pixelSize: 10 }
            Label { text: "MaxWin " + (row.max_win || 0) + "%"; color: "#B8E6CB"; font.pixelSize: 10 }
            Label { text: "MaxLoss " + (row.max_loss || 0) + "%"; color: "#F3C0C0"; font.pixelSize: 10 }
            Label { text: (row.trade_distribution || "W0/L0"); color: "#AFC3DF"; font.pixelSize: 10 }
        }
        Label {
            text: row.explanation || ""
            color: "#98B4D4"
            font.pixelSize: 10
            wrapMode: Text.Wrap
            maximumLineCount: 2
            elide: Text.ElideRight
            Layout.fillWidth: true
        }
        Label {
            text: row.performance_context || ""
            color: "#89C7BE"
            font.pixelSize: 10
            wrapMode: Text.Wrap
            maximumLineCount: 1
            elide: Text.ElideRight
            Layout.fillWidth: true
        }
        RowLayout {
            Label { text: "CtxConf " + Math.round((row.context_confidence || 0) * 100) + "%"; color: "#8ED2D7"; font.pixelSize: 10 }
            Label { text: "Robust " + (row.behavior_robustness || 0); color: "#D8CF8E"; font.pixelSize: 10 }
        }
    }

    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        onEntered: root.hovered = true
        onExited: root.hovered = false
        onClicked: root.clicked()
    }
}
